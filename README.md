# MCT-Self-Refine-MCTSr-is-openai
A simple program of the MCT Self-Refine (MCTSr) algorithm, welcome to adapt or modify it
# how to use is program?
OpenAI's API is used, and koboldcpp is used by default, which proves that the local model can also solve Mathematical Olympiad puzzles.
Modify the following paragraphs according to your needs:
client = OpenAI(
    api_key="...",
    base_url="http://127.0.0.1:5001/v1",
)
and 
Use Notepad's one-click replacement function to replace this sentence "You are a coding assistant who can complete programs independently." or "You are an assistant helping with mathematical reasoning."
# python dependencies
pip install PyQt5 numpy openai PyMuPDF chromadb transformers sentence-transformers treelib

# This is the original paper, of which I am not the author.

https://arxiv.org/abs/2406.07394

# py file with the word coding (for writing code)
# py file without the word coding (for math)

