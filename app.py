from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_model import generate_answer

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Define a route for the root URL
@app.route("/", methods=["GET"])
def home():
    return "<h1>Welcome to the RAG-based QA Bot API!</h1>"

# Flask route for the RAG-based QA bot
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "No question provided."}), 400
    
    # Generate RAG-based answer
    answer = generate_answer(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import openai
# import pinecone
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize Pinecone
# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
# index_name = "qa-bot"
# index = pinecone.Index(index_name)

# # Flask app setup
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#     # Parse question from request
#     data = request.json
#     question = data.get("question", "")

#     # Use OpenAI to generate the query embedding
#     question_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']

#     # Query Pinecone to find relevant documents
#     query_results = index.query(queries=[question_embedding], top_k=3, include_metadata=True)

#     # Get the top document text as context
#     context = "\n\n".join([match["metadata"]["text"] for match in query_results["matches"]])

#     # Use OpenAI to generate an answer based on the context
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=f"Answer the following question using the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:",
#         temperature=0.7,
#         max_tokens=150
#     )

#     # Extract the answer from OpenAI's response
#     answer = response["choices"][0]["text"].strip()

#     # Return the answer as a JSON response
#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(debug=True)
