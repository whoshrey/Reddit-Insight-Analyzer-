import streamlit as st
import praw
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import streamlit as st
import praw

client_id = st.secrets["REDDIT_CLIENT_ID"]
client_secret = st.secrets["REDDIT_CLIENT_SECRET"]
# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="reddit-insight-analyzer"
)
# Load models
from transformers import pipeline

from transformers import pipeline

abuse_detector = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=-1  # ← forces CPU
)

emotion_detector = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1  # ← forces CPU
)

# Streamlit UI
st.title("🔍 Reddit Insight Analyzer")

# Sidebar filters
subreddit_name = st.text_input("Enter Subreddit:", "AskReddit")
post_filter = st.selectbox("Choose Post Filter:", ["hot", "top", "new", "rising"])
time_filter = st.selectbox("Choose Time Filter (for 'top'):", ["day", "week", "month", "year", "all"])
limit = st.slider("Number of Posts", 1, 10, 3)

# Fetch posts
def fetch_posts():
    subreddit = reddit.subreddit(subreddit_name)
    if post_filter == "hot":
        posts = subreddit.hot(limit=limit)
    elif post_filter == "top":
        posts = subreddit.top(time_filter=time_filter, limit=limit)
    elif post_filter == "new":
        posts = subreddit.new(limit=limit)
    elif post_filter == "rising":
        posts = subreddit.rising(limit=limit)
    else:
        posts = []

    post_data = []
    for post in posts:
        post.comments.replace_more(limit=0)
        comments = [comment.body for comment in post.comments.list()]
        post_data.append({
            "title": post.title,
            "comments": comments,
            "num_comments": len(comments)
        })
    return post_data

if st.button("Analyze"):
    posts = fetch_posts()
    for idx, post in enumerate(posts, start=1):
        st.subheader(f"Post {idx}: {post['title']}")
        st.write(f"Number of Comments: {post['num_comments']}")

        # Abusive Comments
        abusive_comments = []
        for c in post["comments"]:
            result = abuse_detector(c[:512])[0]
            if result['label'] == 'toxic' and result['score'] > 0.9:
                abusive_comments.append((c, result['score']))
        st.markdown("**Abusive Comments:**")
        if abusive_comments:
            for comment, score in abusive_comments:
                st.write(f"⚠️ {score:.2f} → {comment}")
        else:
            st.write("✅ None detected.")

        # Emotion Analysis
        emotion_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for comment in post["comments"]:
            scores = emotion_detector(comment[:512])
            if scores and isinstance(scores, list) and isinstance(scores[0], list):
                top_emotion = max(scores[0], key=lambda x: x["score"])
                label = top_emotion["label"].lower()
                if label == "joy":
                    emotion_counts["positive"] += 1
                elif label in ["anger", "sadness", "fear", "disgust"]:
                    emotion_counts["negative"] += 1
                else:
                    emotion_counts["neutral"] += 1
        st.write(f"😊 Positive: {emotion_counts['positive']} | 😠 Negative: {emotion_counts['negative']} | 😐 Neutral: {emotion_counts['neutral']}")

        # WordCloud
        comment_text = " ".join(post["comments"]).strip()
        if comment_text:
            wc = WordCloud(width=800, height=400, background_color='white',
                           stopwords=set(STOPWORDS)).generate(comment_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Word Cloud for Post {idx}")
            plt.tight_layout()
            st.pyplot(fig)  # pass the figure here




