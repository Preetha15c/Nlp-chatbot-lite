# Offline Chatbot (No TorchText)

A simple, offline-capable chatbot built with Python and PyTorch, without using TorchText. This project demonstrates training and inference of a Seq2Seq-based chatbot with minimal dependencies.

---

## 📁 Project Structure

```
offline_chatbot_no_torchtext/
├── model.py              # Defines the seq2seq model architecture
├── train.py              # Script to train the chatbot model
├── infer.py              # Script for running inference (chat interface)
│   ├── conversations.txt # Sample conversation data used for training
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/offline-chatbot-no-torchtext.git
cd offline-chatbot-no-torchtext
```

### 2. Install Requirements

```bash
pip install torch numpy
```

(Adjust dependencies as needed)

---

## 🧠 How It Works

- `train.py`: Trains a basic sequence-to-sequence model on simple conversation data.
- `infer.py`: Loads the trained model and allows offline chatting in the console.
- `model.py`: Contains encoder-decoder architecture for chatbot training.

---

## 📦 Dependencies

- Python 3.6+
- PyTorch
- NumPy

---

## 🗃️ Dataset

Uses a small custom dataset (`conversations.txt`) consisting of question-answer pairs.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests. Contributions are welcome!
