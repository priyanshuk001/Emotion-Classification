import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class EmotionClassifier:
    def __init__(self, model_path, scaler_path, encoder_path):
        """Initialize the emotion classifier with trained model and preprocessors"""
        try:
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)
            self.loaded_successfully = True
        except Exception as e:
            st.error(f"Error loading model components: {str(e)}")
            self.loaded_successfully = False
            raise e

    def zcr(self, data, frame_length, hop_length):
        """Extract Zero Crossing Rate feature"""
        zcr_feature = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(zcr_feature)

    def rmse(self, data, frame_length=2048, hop_length=512):
        """Extract Root Mean Square Energy feature"""
        rmse_feature = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(rmse_feature)

    def mfcc(self, data, sr, frame_length=2048, hop_length=512, flatten=True):
        """Extract MFCC features"""
        mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
        return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

    def extract_features(self, data, sr=22050, frame_length=2048, hop_length=512):
        """Extract comprehensive features from audio data"""
        result = np.array([])
        
        result = np.hstack((result,
                          self.zcr(data, frame_length, hop_length),
                          self.rmse(data, frame_length, hop_length),
                          self.mfcc(data, sr, frame_length, hop_length)
                         ))
        return result

    def get_predict_feat(self, audio_data, sr):
        """Process audio data and extract features for prediction"""
        try:
            # Use the provided audio data directly
            res = self.extract_features(audio_data, sr)
            
            # Ensure consistent feature length
            desired_length = 2376
            current_length = len(res)
            
            if current_length > desired_length:
                res = res[:desired_length]
            elif current_length < desired_length:
                res = np.pad(res, (0, desired_length - current_length), 'constant')
            
            # Reshape and scale
            result = np.reshape(res, (1, desired_length))
            i_result = self.scaler.transform(result)
            final_result = np.expand_dims(i_result, axis=2)
            
            return final_result
            
        except Exception as e:
            st.error(f"Error processing audio data: {str(e)}")
            return None

    def prediction(self, audio_data, sr):
        """Predict emotion from audio data"""
        res = self.get_predict_feat(audio_data, sr)
        
        if res is None:
            return None, None, None
            
        # Get predictions
        predictions = self.model.predict(res, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        confidence_scores = predictions[0]
        
        # Get emotion labels
        dummy_encoded = np.eye(len(predictions[0]))
        all_emotions = self.encoder.inverse_transform(dummy_encoded)
        all_emotions = [emotion[0] for emotion in all_emotions]
        
        predicted_emotion = all_emotions[predicted_class_idx]
        
        return predicted_emotion, confidence_scores, all_emotions

def load_audio(uploaded_file):
    """Load audio file from Streamlit upload"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load audio with librosa
        audio_data, sr = librosa.load(tmp_path, duration=2.5, offset=0.6)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return audio_data, sr
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def create_waveform_plot(audio_data, sr):
    """Create waveform visualization"""
    time = np.linspace(0, len(audio_data)/sr, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', name='Waveform', line=dict(color='#1f77b4')))
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=300,
        showlegend=False
    )
    return fig

def create_confidence_chart(emotions, confidences):
    """Create confidence scores visualization"""
    # Sort emotions by confidence for better visualization
    sorted_data = sorted(zip(emotions, confidences), key=lambda x: x[1], reverse=True)
    sorted_emotions, sorted_confidences = zip(*sorted_data)
    
    # Create color scale - highest confidence gets strongest color
    colors = px.colors.sequential.Viridis[::-1][:len(emotions)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(sorted_emotions),
            y=list(sorted_confidences),
            marker_color=colors,
            text=[f'{conf:.3f}' for conf in sorted_confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Emotion Confidence Scores',
        xaxis_title='Emotions',
        yaxis_title='Confidence Score',
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_pie_chart(emotions, confidences):
    """Create pie chart for confidence distribution"""
    fig = go.Figure(data=[
        go.Pie(
            labels=emotions,
            values=confidences,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Emotion Confidence Distribution',
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="🎭 Emotion Classification from Audio",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎭 Emotion Classification from Audio")
    st.markdown("Upload audio file(s) to analyze the emotional content using deep learning!")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Input fields for model file paths
        st.markdown("**Enter paths to your model files:**")
        model_path = st.text_input(
            "Model File Path (.keras)", 
            value="cnn_model_full.keras",
            help="Path to your trained model file"
        )
        scaler_path = st.text_input(
            "Scaler File Path (.pkl)", 
            value="emotion_scaler.pkl",
            help="Path to your scaler file"
        )
        encoder_path = st.text_input(
            "Encoder File Path (.pkl)", 
            value="emotion_encoder.pkl",
            help="Path to your encoder file"
        )
        
        # Button to load model
        if st.button("🔄 Load Model", type="primary"):
            # Check if files exist
            missing_files = []
            if not os.path.exists(model_path):
                missing_files.append(f"Model: {model_path}")
            if not os.path.exists(scaler_path):
                missing_files.append(f"Scaler: {scaler_path}")
            if not os.path.exists(encoder_path):
                missing_files.append(f"Encoder: {encoder_path}")
            
            if missing_files:
                st.error("❌ Missing files:")
                for file in missing_files:
                    st.error(f"  • {file}")
            else:
                # Initialize classifier
                with st.spinner("Loading model..."):
                    try:
                        classifier = EmotionClassifier(model_path, scaler_path, encoder_path)
                        st.session_state.classifier = classifier
                        st.success("✅ Model loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
        
        # Auto-load model if files exist and haven't been loaded yet
        if 'classifier' not in st.session_state:
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
                with st.spinner("Auto-loading model..."):
                    try:
                        classifier = EmotionClassifier(model_path, scaler_path, encoder_path)
                        st.session_state.classifier = classifier
                        st.success("✅ Model auto-loaded successfully!")
                    except Exception as e:
                        st.warning(f"⚠️ Auto-load failed: {str(e)}")
                        st.info("Please click 'Load Model' button to try again")
        
        # Show model status
        if 'classifier' in st.session_state:
            st.success("🟢 Model Ready")
        else:
            st.error("🔴 Model Not Loaded")
            
        # Model info
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.info("""
        • Place your model files in the same folder as this app
        • Default names: `cnn_model_full.keras`, `emotion_scaler.pkl`, `emotion_encoder.pkl`
        • Use full paths if files are in different locations
        """)
    
    # Main content area
    if 'classifier' in st.session_state:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📁 Upload Audio")
            
            # Upload mode selection
            upload_mode = st.radio(
                "Select upload mode:",
                ["Single File", "Multiple Files"],
                horizontal=True
            )
            
            if upload_mode == "Single File":
                uploaded_file = st.file_uploader(
                    "Choose an audio file", 
                    type=['wav', 'mp3', 'flac', 'm4a'],
                    help="Upload an audio file for emotion analysis"
                )
                
                if uploaded_file is not None:
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    
                    # Display file info
                    st.info(f"**File size:** {uploaded_file.size:,} bytes")
                    
                    # Audio player
                    st.audio(uploaded_file)
                    
                    # Process button
                    if st.button("🔍 Analyze Emotion", type="primary"):
                        with st.spinner("Processing audio..."):
                            # Load and process audio
                            audio_data, sr = load_audio(uploaded_file)
                            
                            if audio_data is not None:
                                # Get prediction
                                predicted_emotion, confidence_scores, all_emotions = st.session_state.classifier.prediction(audio_data, sr)
                                
                                if predicted_emotion is not None:
                                    # Store results in session state
                                    st.session_state.results = {
                                        'mode': 'single',
                                        'predicted_emotion': predicted_emotion,
                                        'confidence_scores': confidence_scores,
                                        'all_emotions': all_emotions,
                                        'audio_data': audio_data,
                                        'sr': sr,
                                        'filename': uploaded_file.name
                                    }
                                    st.success("✅ Analysis complete!")
                                else:
                                    st.error("❌ Failed to process audio file")
            
            else:  # Multiple Files
                uploaded_files = st.file_uploader(
                    "Choose audio files", 
                    type=['wav', 'mp3', 'flac', 'm4a'],
                    accept_multiple_files=True,
                    help="Upload multiple audio files for batch emotion analysis"
                )
                
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} files uploaded")
                    
                    # Display file info
                    total_size = sum(f.size for f in uploaded_files)
                    st.info(f"**Total files:** {len(uploaded_files)} | **Total size:** {total_size:,} bytes")
                    
                    # Show file list
                    with st.expander("📋 Uploaded Files", expanded=False):
                        for i, file in enumerate(uploaded_files, 1):
                            st.write(f"{i}. {file.name} ({file.size:,} bytes)")
                    
                    # Process button
                    if st.button("🔍 Analyze All Files", type="primary"):
                        batch_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            try:
                                # Load and process audio
                                audio_data, sr = load_audio(uploaded_file)
                                
                                if audio_data is not None:
                                    # Get prediction
                                    predicted_emotion, confidence_scores, all_emotions = st.session_state.classifier.prediction(audio_data, sr)
                                    
                                    if predicted_emotion is not None:
                                        batch_results.append({
                                            'filename': uploaded_file.name,
                                            'predicted_emotion': predicted_emotion,
                                            'confidence_scores': confidence_scores,
                                            'all_emotions': all_emotions,
                                            'max_confidence': np.max(confidence_scores)
                                        })
                                    else:
                                        batch_results.append({
                                            'filename': uploaded_file.name,
                                            'predicted_emotion': 'Error',
                                            'confidence_scores': None,
                                            'all_emotions': None,
                                            'max_confidence': 0
                                        })
                                else:
                                    batch_results.append({
                                        'filename': uploaded_file.name,
                                        'predicted_emotion': 'Error',
                                        'confidence_scores': None,
                                        'all_emotions': None,
                                        'max_confidence': 0
                                    })
                            except Exception as e:
                                st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                                batch_results.append({
                                    'filename': uploaded_file.name,
                                    'predicted_emotion': 'Error',
                                    'confidence_scores': None,
                                    'all_emotions': None,
                                    'max_confidence': 0
                                })
                        
                        # Store batch results
                        st.session_state.results = {
                            'mode': 'batch',
                            'batch_results': batch_results
                        }
                        
                        status_text.text("✅ Batch analysis complete!")
                        progress_bar.empty()
        
        with col2:
            st.header("📊 Results")
            
            if 'results' in st.session_state:
                results = st.session_state.results
                
                if results['mode'] == 'single':
                    # Single file results
                    # Main prediction result
                    st.markdown("### 🎯 Prediction")
                    max_confidence = np.max(results['confidence_scores'])
                    
                    # Create a nice result card
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        color: white;
                        margin: 10px 0;
                    ">
                        <h2 style="margin: 0; color: white;">{results['predicted_emotion'].upper()}</h2>
                        <p style="margin: 5px 0; font-size: 18px;">Confidence: {max_confidence:.4f} ({max_confidence*100:.2f}%)</p>
                        <p style="margin: 5px 0; font-size: 14px;">File: {results['filename']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence scores table
                    st.markdown("### 📈 Detailed Scores")
                    scores_df = pd.DataFrame({
                        'Emotion': results['all_emotions'],
                        'Confidence': results['confidence_scores'],
                        'Percentage': [f"{score*100:.2f}%" for score in results['confidence_scores']]
                    }).sort_values('Confidence', ascending=False)
                    
                    st.dataframe(scores_df, use_container_width=True, hide_index=True)
                
                else:  # Batch results
                    st.markdown("### 🎯 Batch Results Summary")
                    
                    # Overall statistics
                    successful_predictions = [r for r in results['batch_results'] if r['predicted_emotion'] != 'Error']
                    total_files = len(results['batch_results'])
                    success_rate = len(successful_predictions) / total_files * 100
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total Files", total_files)
                    with col_stat2:
                        st.metric("Successful", len(successful_predictions))
                    with col_stat3:
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                    # Emotion distribution
                    if successful_predictions:
                        emotion_counts = {}
                        for result in successful_predictions:
                            emotion = result['predicted_emotion']
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        st.markdown("### 📊 Emotion Distribution")
                        dist_df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count'])
                        dist_df['Percentage'] = (dist_df['Count'] / len(successful_predictions) * 100).round(1)
                        st.dataframe(dist_df, use_container_width=True, hide_index=True)
                    
                    # Detailed results table
                    st.markdown("### 📋 Detailed Results")
                    
                    # Create comprehensive results dataframe
                    detailed_results = []
                    for result in results['batch_results']:
                        row = {
                            'Filename': result['filename'],
                            'Predicted Emotion': result['predicted_emotion'],
                            'Max Confidence': f"{result['max_confidence']:.4f}" if result['max_confidence'] > 0 else "N/A"
                        }
                        
                        # Add individual emotion confidences
                        if result['confidence_scores'] is not None and result['all_emotions'] is not None:
                            for i, emotion in enumerate(result['all_emotions']):
                                row[f'{emotion}_conf'] = f"{result['confidence_scores'][i]:.3f}"
                        
                        detailed_results.append(row)
                    
                    detailed_df = pd.DataFrame(detailed_results)
                    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    # Download results button
                    csv = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="emotion_analysis_results.csv",
                        mime="text/csv"
                    )
                
            else:
                st.info("Upload audio file(s) and click analyze to see results here!")
        
        # Visualizations (full width)
        if 'results' in st.session_state:
            st.markdown("---")
            st.header("📊 Visualizations")
            
            results = st.session_state.results
            
            if results['mode'] == 'single':
                # Single file visualizations
                # Two columns for charts
                vis_col1, vis_col2 = st.columns(2)
                
                with vis_col1:
                    # Waveform plot
                    waveform_fig = create_waveform_plot(results['audio_data'], results['sr'])
                    st.plotly_chart(waveform_fig, use_container_width=True)
                    
                    # Confidence bar chart
                    confidence_fig = create_confidence_chart(results['all_emotions'], results['confidence_scores'])
                    st.plotly_chart(confidence_fig, use_container_width=True)
                
                with vis_col2:
                    # Confidence pie chart
                    pie_fig = create_confidence_pie_chart(results['all_emotions'], results['confidence_scores'])
                    st.plotly_chart(pie_fig, use_container_width=True)
                    
                    # Feature information
                    st.markdown("### 🔍 Audio Features")
                    st.info(f"""
                    **Sample Rate:** {results['sr']} Hz  
                    **Duration:** {len(results['audio_data'])/results['sr']:.2f} seconds  
                    **Samples:** {len(results['audio_data']):,}
                    """)
            
            else:  # Batch visualizations
                # Batch analysis visualizations
                successful_predictions = [r for r in results['batch_results'] if r['predicted_emotion'] != 'Error']
                
                if successful_predictions:
                    vis_col1, vis_col2 = st.columns(2)
                    
                    with vis_col1:
                        # Emotion distribution pie chart
                        emotion_counts = {}
                        for result in successful_predictions:
                            emotion = result['predicted_emotion']
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        fig_pie = go.Figure(data=[
                            go.Pie(
                                labels=list(emotion_counts.keys()),
                                values=list(emotion_counts.values()),
                                hole=0.3,
                                textinfo='label+percent',
                                textposition='outside'
                            )
                        ])
                        fig_pie.update_layout(title='Emotion Distribution Across All Files', height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with vis_col2:
                        # Confidence distribution histogram
                        confidences = [r['max_confidence'] for r in successful_predictions]
                        
                        fig_hist = go.Figure(data=[
                            go.Histogram(
                                x=confidences,
                                nbinsx=20,
                                marker_color='lightblue',
                                opacity=0.7
                            )
                        ])
                        fig_hist.update_layout(
                            title='Confidence Score Distribution',
                            xaxis_title='Confidence Score',
                            yaxis_title='Number of Files',
                            height=400
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Average confidence by emotion
                    st.markdown("### 📈 Average Confidence by Emotion")
                    emotion_confidence = {}
                    emotion_count = {}
                    
                    for result in successful_predictions:
                        emotion = result['predicted_emotion']
                        confidence = result['max_confidence']
                        
                        if emotion not in emotion_confidence:
                            emotion_confidence[emotion] = 0
                            emotion_count[emotion] = 0
                        
                        emotion_confidence[emotion] += confidence
                        emotion_count[emotion] += 1
                    
                    avg_confidence_data = []
                    for emotion in emotion_confidence:
                        avg_conf = emotion_confidence[emotion] / emotion_count[emotion]
                        avg_confidence_data.append({
                            'Emotion': emotion,
                            'Average Confidence': avg_conf,
                            'Count': emotion_count[emotion]
                        })
                    
                    avg_conf_df = pd.DataFrame(avg_confidence_data).sort_values('Average Confidence', ascending=False)
                    
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=avg_conf_df['Emotion'],
                            y=avg_conf_df['Average Confidence'],
                            marker_color='lightgreen',
                            text=[f'{conf:.3f}' for conf in avg_conf_df['Average Confidence']],
                            textposition='auto'
                        )
                    ])
                    fig_bar.update_layout(
                        title='Average Confidence Score by Emotion',
                        xaxis_title='Emotion',
                        yaxis_title='Average Confidence',
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("👈 Please load your model files in the sidebar to get started!")
        
        # Instructions
        st.markdown("""
        ## 🚀 How to Use
        
        1. **Set Model File Paths** (in sidebar):
           - Enter paths to your model files
           - Default paths assume files are in the same folder
           - Click "Load Model" or files will auto-load if they exist
        
        2. **Upload Audio File(s)**:
           - Choose between Single File or Multiple Files mode
           - Supported formats: WAV, MP3, FLAC, M4A
           - The app will analyze a 2.5-second segment from each file
        
        3. **Analyze**:
           - Click the "Analyze Emotion" (single) or "Analyze All Files" (batch) button
           - View results and visualizations
        
        ## 📋 Features
        
        - **Single & Batch Processing**: Analyze one file or multiple files at once
        - **Easy Setup**: Just specify file paths, no need to upload model files
        - **Auto-Loading**: Model loads automatically if files are found
        - **Real-time Analysis**: Get instant emotion predictions
        - **Interactive Visualizations**: Waveform, confidence charts, and pie charts
        - **Detailed Results**: See confidence scores for all emotions
        - **CSV Export**: Download batch results as CSV file
        - **Progress Tracking**: Visual progress bar for batch processing
        - **Responsive Design**: Works on desktop and mobile
        
        ## 📁 File Setup
        
        Place your model files in the same folder as this app:
        - `cnn_model_full.keras`
        - `emotion_scaler.pkl`
        - `emotion_encoder.pkl`
        
        Or specify full paths in the sidebar if they're located elsewhere.
        """)

if __name__ == "__main__":
    main()