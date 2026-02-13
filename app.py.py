import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('ticket_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    categories = joblib.load('categories.pkl')
    return model, vectorizer, categories

model, vectorizer, categories = load_models()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# App title
st.title('🎫 Support Ticket Classifier')
st.write('Automatically categorize support tickets using AI')

st.markdown('---')

# Input methods
st.subheader('Enter Support Ticket')

input_method = st.radio('Choose input method:', ['Type Text', 'Use Example'])

if input_method == 'Type Text':
    ticket_text = st.text_area(
        'Ticket Description:', 
        height=150,
        placeholder='Example: I need access to the HR system to view my payroll information...'
    )
else:
    examples = {
        'Hardware Issue': 'My laptop screen is flickering and I cannot connect to the monitor. Need urgent hardware support.',
        'HR Support': 'I need to update my personal information and access my payroll records for tax purposes.',
        'Access Request': 'Please grant me access to the project management tool and shared drive for the new project.',
        'Storage Request': 'I need additional cloud storage space. My current allocation is full.',
        'Purchase Request': 'Need approval to purchase new software licenses for the design team.'
    }
    
    selected_example = st.selectbox('Select an example:', list(examples.keys()))
    ticket_text = examples[selected_example]
    st.text_area('Ticket Description:', value=ticket_text, height=100, disabled=True)

st.markdown('---')

# Predict button
if st.button('🔮 Classify Ticket', type='primary'):
    if ticket_text and len(ticket_text.strip()) > 0:
        # Clean text
        cleaned = clean_text(ticket_text)
        
        # Vectorize
        text_vec = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        
        # Get top 3 predictions
        top_3_idx = probabilities.argsort()[-3:][::-1]
        
        # Display results
        st.subheader('Classification Results')
        
        # Main prediction
        st.success(f'**Primary Category:** {prediction}')
        
        confidence = probabilities.max()
        st.metric('Confidence', f'{confidence*100:.1f}%')
        
        # Progress bar for confidence
        st.progress(confidence)
        
        # Top 3 predictions
        st.markdown('---')
        st.subheader('Top 3 Predictions')
        
        for idx in top_3_idx:
            category = categories[idx]
            prob = probabilities[idx]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f'**{category}**')
            with col2:
                st.write(f'{prob*100:.1f}%')
            st.progress(prob)
        
    else:
        st.warning('⚠️ Please enter a ticket description')

# Sidebar info
st.sidebar.title('About')
st.sidebar.info(
    'This app uses a Logistic Regression model trained on 47,837 support tickets '
    'to automatically categorize incoming requests.'
)

st.sidebar.markdown('---')
st.sidebar.subheader('Model Details')
st.sidebar.write('**Algorithm:** Logistic Regression')
st.sidebar.write('**Accuracy:** 85.38%')
st.sidebar.write('**Features:** TF-IDF (5,000)')
st.sidebar.write('**Categories:** 8')

st.sidebar.markdown('---')
st.sidebar.subheader('Categories')
for cat in sorted(categories):
    st.sidebar.write(f'• {cat}')

st.sidebar.markdown('---')
st.sidebar.caption('Built by [Your Name]')
st.sidebar.caption('[GitHub](https://github.com/e-ric79)')