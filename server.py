from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from emergentintegrations.llm.chat import LlmChat, UserMessage


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pinned: bool = False
    is_public: bool = False
    share_token: Optional[str] = None
    saved: bool = False
    tags: List[str] = []
    importance: str = "medium"  # low, medium, high

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    saved: Optional[bool] = None
    tags: Optional[List[str]] = None
    importance: Optional[str] = None

class ShareResponse(BaseModel):
    share_url: str
    share_token: str


# Chat routes
@api_router.post("/conversations", response_model=Conversation)
async def create_conversation(input: ConversationCreate):
    """Create a new conversation"""
    conversation = Conversation(title=input.title)
    doc = conversation.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    await db.conversations.insert_one(doc)
    return conversation

@api_router.get("/conversations", response_model=List[Conversation])
async def get_conversations(
    saved_only: bool = False,
    search: Optional[str] = None,
    tag: Optional[str] = None,
    importance: Optional[str] = None,
    sort_by: str = "updated_at"  # updated_at, created_at, title, importance
):
    """Get conversations with optional filtering and sorting"""
    query = {}
    
    # Filter by saved status
    if saved_only:
        query["saved"] = True
    
    # Filter by tag
    if tag:
        query["tags"] = tag
    
    # Filter by importance
    if importance:
        query["importance"] = importance
    
    conversations = await db.conversations.find(query, {"_id": 0}).to_list(1000)
    
    for conv in conversations:
        if isinstance(conv['created_at'], str):
            conv['created_at'] = datetime.fromisoformat(conv['created_at'])
        if isinstance(conv['updated_at'], str):
            conv['updated_at'] = datetime.fromisoformat(conv['updated_at'])
        # Set default values for new fields if not present
        if 'pinned' not in conv:
            conv['pinned'] = False
        if 'is_public' not in conv:
            conv['is_public'] = False
        if 'share_token' not in conv:
            conv['share_token'] = None
        if 'saved' not in conv:
            conv['saved'] = False
        if 'tags' not in conv:
            conv['tags'] = []
        if 'importance' not in conv:
            conv['importance'] = 'medium'
    
    # Filter by search text
    if search:
        search_lower = search.lower()
        conversations = [c for c in conversations if search_lower in c['title'].lower()]
    
    # Sort conversations
    if sort_by == "title":
        conversations.sort(key=lambda x: x['title'].lower())
    elif sort_by == "created_at":
        conversations.sort(key=lambda x: -x['created_at'].timestamp())
    elif sort_by == "importance":
        importance_order = {"high": 0, "medium": 1, "low": 2}
        conversations.sort(key=lambda x: importance_order.get(x.get('importance', 'medium'), 1))
    else:  # default: updated_at
        conversations.sort(key=lambda x: -x['updated_at'].timestamp())
    
    # Always put pinned conversations first
    conversations.sort(key=lambda x: not x.get('pinned', False))
    
    return conversations

@api_router.get("/conversations/{conversation_id}/messages", response_model=List[Message])
async def get_messages(conversation_id: str):
    """Get all messages for a conversation"""
    messages = await db.messages.find(
        {"conversation_id": conversation_id}, 
        {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)
    
    for msg in messages:
        if isinstance(msg['timestamp'], str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    
    return messages

@api_router.post("/chat", response_model=Message)
async def chat(request: ChatRequest):
    """Send a message and get AI response"""
    try:
        # Save user message
        user_message = Message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.message
        )
        user_doc = user_message.model_dump()
        user_doc['timestamp'] = user_doc['timestamp'].isoformat()
        await db.messages.insert_one(user_doc)
        
        # Get conversation history
        messages = await db.messages.find(
            {"conversation_id": request.conversation_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(1000)
        
        # Initialize LLM chat
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        chat = LlmChat(
            api_key=api_key,
            session_id=request.conversation_id,
            system_message="You are a helpful AI assistant. Provide clear, concise, and helpful responses."
        ).with_model("openai", "gpt-4o-mini")
        
        # Send message to AI
        llm_message = UserMessage(text=request.message)
        ai_response = await chat.send_message(llm_message)
        
        # Save AI response
        assistant_message = Message(
            conversation_id=request.conversation_id,
            role="assistant",
            content=ai_response
        )
        assistant_doc = assistant_message.model_dump()
        assistant_doc['timestamp'] = assistant_doc['timestamp'].isoformat()
        await db.messages.insert_one(assistant_doc)
        
        # Update conversation updated_at
        await db.conversations.update_one(
            {"id": request.conversation_id},
            {"$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        return assistant_message
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@api_router.get("/tags")
async def get_all_tags():
    """Get all unique tags from conversations"""
    try:
        conversations = await db.conversations.find({}, {"_id": 0, "tags": 1}).to_list(1000)
        all_tags = set()
        for conv in conversations:
            if 'tags' in conv:
                all_tags.update(conv['tags'])
        return {"tags": sorted(list(all_tags))}
    except Exception as e:
        logger.error(f"Get tags error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@api_router.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, update: ConversationUpdate):
    """Update conversation (title, pin, save, tags, importance)"""
    try:
        update_data = {}
        if update.title is not None:
            update_data["title"] = update.title
        if update.pinned is not None:
            update_data["pinned"] = update.pinned
        if update.saved is not None:
            update_data["saved"] = update.saved
        if update.tags is not None:
            update_data["tags"] = update.tags
        if update.importance is not None:
            update_data["importance"] = update.importance
        
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        result = await db.conversations.update_one(
            {"id": conversation_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get updated conversation
        conv = await db.conversations.find_one({"id": conversation_id}, {"_id": 0})
        if conv:
            if isinstance(conv['created_at'], str):
                conv['created_at'] = datetime.fromisoformat(conv['created_at'])
            if isinstance(conv['updated_at'], str):
                conv['updated_at'] = datetime.fromisoformat(conv['updated_at'])
        
        return conv
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")

@api_router.post("/conversations/{conversation_id}/share", response_model=ShareResponse)
async def share_conversation(conversation_id: str):
    """Generate a shareable link for the conversation"""
    try:
        # Generate unique share token
        share_token = str(uuid.uuid4())
        
        # Update conversation with share token
        result = await db.conversations.update_one(
            {"id": conversation_id},
            {"$set": {"is_public": True, "share_token": share_token}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Generate share URL (you can customize this)
        share_url = f"/shared/{share_token}"
        
        return ShareResponse(share_url=share_url, share_token=share_token)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Share error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Share error: {str(e)}")

@api_router.get("/shared/{share_token}")
async def get_shared_conversation(share_token: str):
    """Get a shared conversation and its messages"""
    try:
        # Find conversation by share token
        conv = await db.conversations.find_one(
            {"share_token": share_token, "is_public": True},
            {"_id": 0}
        )
        
        if not conv:
            raise HTTPException(status_code=404, detail="Shared conversation not found")
        
        # Get messages
        messages = await db.messages.find(
            {"conversation_id": conv['id']},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(1000)
        
        # Convert timestamps
        if isinstance(conv['created_at'], str):
            conv['created_at'] = datetime.fromisoformat(conv['created_at'])
        if isinstance(conv['updated_at'], str):
            conv['updated_at'] = datetime.fromisoformat(conv['updated_at'])
        
        for msg in messages:
            if isinstance(msg['timestamp'], str):
                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
        
        return {
            "conversation": conv,
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get shared conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    try:
        # Delete all messages
        await db.messages.delete_many({"conversation_id": conversation_id})
        
        # Delete conversation
        result = await db.conversations.delete_one({"id": conversation_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


# Old status routes
@api_router.get("/")
async def root():
    return {"message": "AI Chatbox API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()