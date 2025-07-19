# 🔍 Reddit Insight Analyzer

A powerful, interactive Streamlit app that lets you explore and analyze Reddit comments using AI and NLP techniques.

---

## 📌 Features

- 🔥 Fetch posts by **Hot**, **Top**, **New**, or **Rising**
- 🕒 Supports time filters: **Day**, **Week**, **Month**, **Year**, **All time**
- 💬 Extracts and displays Reddit post titles and comment threads
- 🧠 Detects **emotional tone** of comments (Positive, Negative, Neutral)
- 🚫 Flags **toxic or abusive comments** using a fine-tuned BERT model
- ☁️ Generates **word clouds** of comments for each post
- ✅ Simple, intuitive UI powered by Streamlit

---

## ⚙️ Tech Stack

- **Python 3**
- **Streamlit** – for building the interactive web UI
- **PRAW** – to fetch posts and comments from Reddit
- **Transformers (HuggingFace)** – for emotion and toxicity detection
- **Matplotlib & WordCloud** – for comment visualization

---

## 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/whoshrey/reddit-insight-analyzer.git
cd reddit-insight-analyzer
```

2. **Creating/Setup Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

---
## 💻 Demo (Coming Soon)

> Live demo will be available on **Streamlit Cloud** 
> _Want help deploying? Let me know._

---

## 🧠 Future Roadmap

⏳ Improve subreddit error handling and robustness

🗳️ Add voting score and post metadata visualization

📊 Visual summary dashboard (pie chart for sentiment etc.)

🌍 Multi-language support for non-English subreddits

💾 Option to export comments or analysis results as CSV

📱 Make mobile-friendly version

## 👨‍💻 Author

**Shreya Jha**
🧑‍🎓 B.Tech, Artificial Intelligence & Machine Learning, GGSIPU 
🚀  AI/ML & Automation Enthusiast  
📫 [Email](mailto:shreyaworks1212@gmail.com) | [LinkedIn](https://linkedin.com/in/shreya-jha-) | [GitHub](https://github.com/whoshrey)

---

## 📄 License

This project is licensed under the MIT License — feel free to fork, use, or contribute!

---

## ⭐️ Support

If you like this project, leave a **star ⭐** on the repo to support it!