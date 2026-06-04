from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chatbot_core as core


@asynccontextmanager
async def lifespan(app: FastAPI):
    core.load_resources()
    print("Modèle chargé et serveur API prêt.")
    yield


app = FastAPI(title="ESIC Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    message: str
    user_id: str = "guest"


@app.post("/predict_chatbot/")
def predict_intent(query: Query):
    ints = core.predict_class(query.message)
    response_text = core.get_response(ints)
    return {
        "message": query.message,
        "response": response_text,
        "intent": ints[0]['intent'] if ints else "fallback",
    }


if __name__ == "__main__":
    print("Lancez le serveur avec : uvicorn api_server:app --reload")
