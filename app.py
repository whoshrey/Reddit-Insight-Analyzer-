import streamlit as st
import praw
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Use Streamlit secrets for deployment, fallback to environment variables for local development
try:
    client_id = st.secrets["REDDIT_CLIENT_ID"]
    client_secret = st.secrets["REDDIT_CLIENT_SECRET"]
    user_agent = st.secrets.get("REDDIT_USER_AGENT", "reddit-analyzer-app/1.0")
except:
    # Fallback to environment variables for local development
    import os
    from dotenv import load_dotenv
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "reddit-analyzer-app/1.0")

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Load ML models with caching for better performance
@st.cache_resource
def load_models():
    abuse_detector = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1  # forces CPU
    )
    
    emotion_detector = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,  # Updated from return_all_scores=True
        device=-1  # forces CPU
    )
    
    return abuse_detector, emotion_detector

# Load models
abuse_detector, emotion_detector = load_models()

# Streamlit UI
st.title("🔍 Reddit Insight Analyzer")
st.markdown("Analyze Reddit posts for sentiment, toxicity, and generate word clouds!")

# Check if credentials are available
if not all([client_id, client_secret, user_agent]):
    st.error("❌ Reddit API credentials are missing!")
    st.info("Please configure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in Streamlit secrets.")
    st.stop()

# Sidebar for configuration
st.sidebar.header("Configuration")
subreddit_name = st.sidebar.text_input("Enter Subreddit:", "AskReddit")
post_filter = st.sidebar.selectbox("Choose Post Filter:", ["hot", "top", "new", "rising"])

# Show time filter only when 'top' is selected
if post_filter == "top":
    time_filter = st.sidebar.selectbox("Choose Time Filter:", ["day", "week", "month", "year", "all"])
else:
    time_filter = "day"  # default value

limit = st.sidebar.slider("Number of Posts", 1, 20, 5)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Start with fewer posts for faster analysis")

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"*Analyzing:* r/{subreddit_name}")
with col2:
    analyze_button = st.button("🚀 Analyze", type="primary")

# Fetch posts function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_posts(subreddit_name, post_filter, time_filter, limit):
    try:
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
            return []

        post_data = []
        for post in posts:
            try:
                # Limit comment loading for performance
                post.comments.replace_more(limit=1)
                comments = [comment.body for comment in post.comments.list()[:50]]  # Limit to 50 comments
                post_data.append({
                    "title": post.title,
                    "score": post.score,
                    "url": post.url,
                    "comments": comments,
                    "num_comments": len(comments)
                })
            except Exception as e:
                st.warning(f"Error processing post: {post.title[:50]}...")
                continue
                
        return post_data
    except Exception as e:
        st.error(f"Error fetching posts from r/{subreddit_name}: {str(e)}")
        return []

# Analysis functions
def analyze_toxicity(comments, abuse_detector):
    """Analyze comments for toxicity"""
    toxic_comments = []
    
    for comment in comments:
        if comment and comment.strip() and len(comment) > 10:  # Skip very short comments
            try:
                result = abuse_detector(comment[:512])[0]  # Limit to 512 chars
                if result['label'] == 'TOXIC' and result['score'] > 0.8:  # Lowered threshold
                    toxic_comments.append((comment[:200], result['score']))  # Limit display length
            except Exception:
                continue
                
    return toxic_comments

def analyze_emotions(comments, emotion_detector):
    """Analyze comments for emotions"""
    emotion_counts = {
        "joy": 0, "anger": 0, "sadness": 0, "fear": 0, 
        "disgust": 0, "surprise": 0, "neutral": 0
    }
    
    for comment in comments:
        if comment and comment.strip() and len(comment) > 10:
            try:
                scores = emotion_detector(comment[:512])
                if scores and isinstance(scores, list) and len(scores) > 0:
                    top_emotion = max(scores, key=lambda x: x["score"])
                    label = top_emotion["label"].lower()
                    if label in emotion_counts:
                        emotion_counts[label] += 1
                    else:
                        emotion_counts["neutral"] += 1
            except Exception:
                emotion_counts["neutral"] += 1
                
    return emotion_counts

def create_wordcloud(comments):
    """Generate word cloud from comments"""
    if not comments:
        return None
        
    # Clean and combine comments
    text = " ".join([c for c in comments if c and c.strip()])
    
    if len(text) < 10:  # Not enough text
        return None
        
    try:
        # Custom stopwords
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(['reddit', 'post', 'comment', 'think', 'people', 'would', 'could', 'really'])
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=custom_stopwords,
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        return wordcloud
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

# Main analysis logic
if analyze_button:
    if not subreddit_name.strip():
        st.error("Please enter a subreddit name!")
    else:
        with st.spinner(f"Fetching posts from r/{subreddit_name}..."):
            posts = fetch_posts(subreddit_name, post_filter, time_filter, limit)
        
        if not posts:
            st.warning("No posts found or error occurred. Try a different subreddit or filter.")
        else:
            st.success(f"✅ Found {len(posts)} posts!")
            
            # Overall statistics
            total_comments = sum(post['num_comments'] for post in posts)
            avg_score = sum(post['score'] for post in posts) / len(posts)
            
            # Display overall stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Posts Analyzed", len(posts))
            with col2:
                st.metric("Total Comments", total_comments)
            with col3:
                st.metric("Avg Post Score", f"{avg_score:.1f}")
            with col4:
                st.metric("Comments/Post", f"{total_comments/len(posts):.1f}")
            
            st.markdown("---")
            
            # Analyze each post
            for idx, post in enumerate(posts, start=1):
                with st.expander(f"📝 Post {idx}: {post['title'][:80]}{'...' if len(post['title']) > 80 else ''}", expanded=(idx == 1)):
                    
                    # Post info
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"*Score:* {post['score']} | *Comments:* {post['num_comments']}")
                    with col2:
                        st.link_button("View on Reddit", post['url'])
                    
                    if not post["comments"]:
                        st.info("No comments found for this post.")
                        continue
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["🚨 Toxicity", "😊 Emotions", "☁ Word Cloud"])
                    
                    with tab1:
                        st.subheader("Toxic Comments Analysis")
                        toxic_comments = analyze_toxicity(post["comments"], abuse_detector)
                        
                        if toxic_comments:
                            st.warning(f"Found {len(toxic_comments)} potentially toxic comments:")
                            for comment, score in toxic_comments:
                                st.write(f"⚠ *Toxicity: {score:.2%}*")
                                st.write(f"> {comment}")
                                st.write("---")
                        else:
                            st.success("✅ No highly toxic comments detected!")
                    
                    with tab2:
                        st.subheader("Emotion Distribution")
                        emotions = analyze_emotions(post["comments"], emotion_detector)
                        
                        # Create emotion visualization
                        emotion_data = {
                            "😊 Joy": emotions["joy"],
                            "😠 Anger": emotions["anger"], 
                            "😢 Sadness": emotions["sadness"],
                            "😨 Fear": emotions["fear"],
                            "🤢 Disgust": emotions["disgust"],
                            "😲 Surprise": emotions["surprise"],
                            "😐 Neutral": emotions["neutral"]
                        }
                        
                        # Display as columns
                        cols = st.columns(len(emotion_data))
                        for i, (emotion, count) in enumerate(emotion_data.items()):
                            with cols[i]:
                                st.metric(emotion, count)
                    
                    with tab3:
                        st.subheader("Word Cloud")
                        wordcloud = create_wordcloud(post["comments"])
                        
                        if wordcloud:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.imshow(wordcloud, interpolation="bilinear")
                            ax.axis("off")
                            ax.set_title(f"Most Common Words - Post {idx}", fontsize=16, pad=20)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("Not enough text to generate a meaningful word cloud.")
                    
                    st.markdown("---")

# Footer
st.markdown("""
---
*About this app:*
- Analyzes Reddit posts for toxicity using BERT models
- Performs emotion analysis on comments
- Generates word clouds from comment text
- Built with Streamlit, PRAW, and Transformers

Note: Analysis results are AI-generated and may not be 100% accurate.
""")



