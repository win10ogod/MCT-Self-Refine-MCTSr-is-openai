import sys
import os
from openai import OpenAI
import fitz  # PyMuPDF
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog)
from PyQt5.QtCore import Qt
from chromadb import Client
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity
from tiktoken import get_encoding
from typing import List, Dict, Any
from treelib import Node, Tree

# Set your OpenAI API key
client = OpenAI(
    api_key="...",
    base_url="http://127.0.0.1:5001/v1",
)


# Initialize ChromaDB
# Initialize ChromaDB
client_chroma = Client(Settings(persist_directory="./chroma_db"))
collection = client_chroma.get_or_create_collection("file_embeddings")
embedding_model = SentenceTransformer("BAAI/bge-m3")


class MCTSelfRefine:
    def __init__(self, model="gpt-4", iterations=15):
        self.model = model
        self.max_tokens = 512
        self.iterations = iterations
        self.tree = Tree()
        self.tree.create_node("Root", "root", data={"answer": "", "score": 0.0, "visits": 0})

    def refine(self, prompt: str) -> str:
        # Check if 'initial' node exists and remove it
        if self.tree.contains("initial"):
            self.tree.remove_node("initial")
        # Initialize the root node with a naive answer
        root_answer = self.generate_initial_answer(prompt)
        self.tree.create_node("Initial Answer", "initial", parent="root",
                              data={"answer": root_answer, "score": 0.0, "visits": 0})

        for _ in range(self.iterations):
            selected_node = self.selection()
            refined_answer = self.self_refine(selected_node.data["answer"])
            refined_score = self.evaluate_answer(prompt, refined_answer)

            new_node_id = f"node_{len(self.tree.nodes)}"
            self.tree.create_node(f"Refined Answer {len(self.tree.nodes)}", new_node_id,
                                  parent=selected_node.identifier,
                                  data={"answer": refined_answer, "score": refined_score, "visits": 1})

            self.backpropagation(new_node_id, refined_score)

        best_node = max(self.tree.children("root"), key=lambda node: node.data["score"])
        return best_node.data["answer"]

    def generate_initial_answer(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "You are an assistant helping with mathematical reasoning."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def selection(self) -> Node:
        """
        Selects the best node for expansion based on UCB formula.
        """
        def ucb_formula(score, parent_visits, node_visits, c=1.4):
            return score + c * np.sqrt(np.log(parent_visits + 1) / (node_visits + 1))

        current_node = self.tree.get_node("root")
        while not current_node.is_leaf():
            children = self.tree.children(current_node.identifier)
            parent_visits = current_node.data["visits"]
            scores = [ucb_formula(child.data["score"], parent_visits, child.data["visits"]) for child in children]
            current_node = children[np.argmax(scores)]

        return current_node

    def self_refine(self, answer: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "You are an assistant helping with mathematical reasoning."},
                {"role": "user", "content": f"Here is an answer: {answer}. Improve and refine this answer."}
            ]
        )
        return response.choices[0].message.content

    def evaluate_answer(self, prompt: str, answer: str) -> float:
        """
        Evaluate the refined answer based on semantic similarity to the question and a predefined rubric.
        """
        rubric = "Rate the correctness and completeness of the answer on a scale of 0 to 10. Be fair but strict."
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "You are an assistant helping with mathematical reasoning."},
                {"role": "user", "content": f"Question: {prompt}\n\nAnswer: {answer}\n\n{rubric}"}
            ]
        )
        try:
            score = float(response.choices[0].message.content.strip())
        except ValueError:
            score = 0.0
        return score

    def backpropagation(self, node_id: str, score: float):
        current_node = self.tree.get_node(node_id)
        while current_node:
            current_node.data["score"] = max(current_node.data["score"], score)
            current_node.data["visits"] += 1
            current_node = self.tree.parent(current_node.identifier)

class ChatInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Chat Interface with OpenAI API")
        self.setGeometry(100, 100, 600, 400)

        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        self.input_area = QTextEdit(self)
        self.input_area.setMaximumHeight(100)
        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        self.upload_button = QPushButton("Upload File", self)
        self.upload_button.clicked.connect(self.upload_file)
        self.export_button = QPushButton("Export Data", self)
        self.export_button.clicked.connect(self.export_to_json)
        layout = QVBoxLayout()
        layout.addWidget(self.chat_area)
        layout.addWidget(self.input_area)
        layout.addWidget(self.send_button)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.export_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize MCT Self-Refine algorithm
        self.mct_sr = MCTSelfRefine()
        self.data_log = []  # Initialize an empty list to store interaction logs

    def send_message(self):
        user_input = self.input_area.toPlainText().strip()
        if not user_input:
            return

        self.append_chat("User", user_input)

        # Handle MCT Self-Refine
        refined_response = self.mct_sr.refine(user_input)
        self.append_chat("Assistant", refined_response)
        interaction_data = {
            "instruction": "You are an assistant helping with mathematical reasoning.",
            "input": user_input,
            "output": refined_response
        }
        self.data_log.append(interaction_data)
        # Clear input area
        self.input_area.clear()
    def export_to_json(self):
        import json
        
        with open("interaction_data.json", "w") as file:
            json.dump(self.data_log, file, indent=4)
        self.append_chat("System", "Data exported to JSON file successfully.")

    def append_chat(self, sender: str, message: str):
        self.chat_area.append(f"<b>{sender}:</b> {message}")

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Text Files (*.txt);;JSON Files (*.json);;Python Files (*.py);;C++ Files (*.cpp);;PDF Files (*.pdf)")
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r") as file:
                text = file.read()

        # Embed and store in ChromaDB
        vectors = self.get_embeddings([text])
        collection.add(documents=[text], embeddings=vectors, metadatas=[{"file_path": file_path}])

        self.append_chat("System", f"File '{file_path}' uploaded and embedded successfully.")

    def extract_text_from_pdf(self, file_path: str) -> str:
        document = fitz.open(file_path)
        text = ""
        for page in document:
            text += page.get_text("text")
        return text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using the Hugging Face model
        """
        embeddings = embedding_model.encode(texts, convert_to_tensor=True).cpu().numpy()
        return embeddings.tolist()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    chat_interface = ChatInterface()
    chat_interface.show()
    sys.exit(app.exec_())