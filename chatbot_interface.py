from chatbot_core import load_resources, predict_class, get_response

load_resources()
print("--- Modèle et données chargés avec succès. Chatbot prêt. ---")
print("--- Démarrez la conversation. Tapez 'quit' pour arrêter. ---\n")

while True:
    message = input("Vous: ")
    if message.lower() == 'quit':
        break
    ints = predict_class(message)
    res = get_response(ints)
    print("ESIC Bot:", res)
