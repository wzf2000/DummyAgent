import json
import logging
import numpy as np
from dataclasses import dataclass
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import torch
import nltk
import os

def ensure_nltk_data():
    """Ensure NLTK data is available"""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logging.warning("VADER lexicon not found, downloading...")
        nltk.download('vader_lexicon', quiet=True)

# Check NLTK data availability at import time
ensure_nltk_data()

@dataclass
class RecommendationMetrics:
    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    top_10_hit_rate: float
    top_15_hit_rate: float
    sim_top_1_hit_rate: float
    sim_top_3_hit_rate: float
    sim_top_5_hit_rate: float
    real_top_1_hit_rate: float
    real_top_3_hit_rate: float
    real_top_5_hit_rate: float
    sim_average_hit_rate: float
    real_average_hit_rate: float
    average_hit_rate: float
    sim_total_scenarios: int
    real_total_scenarios: int
    total_scenarios: int
    sim_top_1_hits: int
    sim_top_3_hits: int
    sim_top_5_hits: int
    real_top_1_hits: int
    real_top_3_hits: int
    real_top_5_hits: int
    top_1_hits: int
    top_3_hits: int
    top_5_hits: int

@dataclass
class SimulationMetrics:
    sim_preference_estimation: float
    real_preference_estimation: float
    preference_estimation: float
    sim_sentiment_error: float
    real_sentiment_error: float
    sentiment_error: float
    sim_emotion_error: float
    real_emotion_error: float
    emotion_error: float
    sim_topic_error: float
    real_topic_error: float
    topic_error: float
    sim_review_generation: float
    real_review_generation: float
    review_generation: float
    sim_overall_quality: float
    real_overall_quality: float
    overall_quality: float

class BaseEvaluator:
    """Base class for evaluation tools"""
    def __init__(self):
        self.metrics_history: list[RecommendationMetrics | SimulationMetrics] = []

    def save_metrics(self, metrics: RecommendationMetrics | SimulationMetrics):
        """Save metrics to history"""
        self.metrics_history.append(metrics)

    def get_metrics_history(self):
        """Get all historical metrics"""
        return self.metrics_history

class RecommendationEvaluator(BaseEvaluator):
    """Evaluator for recommendation tasks"""
    
    def __init__(self):
        super().__init__()
        self.n_values = [1, 3, 5, 10, 15]  # 预定义的n值数组

    def calculate_hr_at_n(
        self,
        ground_truth: list[str],
        predictions: list[list[str]],
        number_sim: int
    ) -> RecommendationMetrics:
        """Calculate Hit Rate at different N values"""
        total = len(ground_truth)
        hits = {n: 0 for n in self.n_values}
        sim_hits = {n: 0 for n in self.n_values}
        real_hits = {n: 0 for n in self.n_values}

        index = 0
        for gt, pred in zip(ground_truth, predictions):
            for n in self.n_values:
                if gt in pred[:n]:
                    hits[n] += 1
                    if index < number_sim:
                        sim_hits[n] += 1
                    else:
                        real_hits[n] += 1
            index += 1

        top_1_hit_rate = hits[1] / total if total > 0 else 0
        top_3_hit_rate = hits[3] / total if total > 0 else 0
        top_5_hit_rate = hits[5] / total if total > 0 else 0
        top_10_hit_rate = hits[10] / total if total > 0 else 0
        top_15_hit_rate = hits[15] / total if total > 0 else 0
        sim_top_1_hit_rate = sim_hits[1] / number_sim if number_sim > 0 else 0
        sim_top_3_hit_rate = sim_hits[3] / number_sim if number_sim > 0 else 0
        sim_top_5_hit_rate = sim_hits[5] / number_sim if number_sim > 0 else 0
        real_top_1_hit_rate = real_hits[1] / (total - number_sim) if total - number_sim > 0 else 0
        real_top_3_hit_rate = real_hits[3] / (total - number_sim) if total - number_sim > 0 else 0
        real_top_5_hit_rate = real_hits[5] / (total - number_sim) if total - number_sim > 0 else 0
        sim_average_hit_rate = (sim_top_1_hit_rate + sim_top_3_hit_rate + sim_top_5_hit_rate) / 3
        real_average_hit_rate = (real_top_1_hit_rate + real_top_3_hit_rate + real_top_5_hit_rate) / 3
        average_hit_rate = (top_1_hit_rate + top_3_hit_rate + top_5_hit_rate) / 3
        metrics = RecommendationMetrics(
            top_1_hit_rate=top_1_hit_rate,
            top_3_hit_rate=top_3_hit_rate,
            top_5_hit_rate=top_5_hit_rate,
            top_10_hit_rate=top_10_hit_rate,
            top_15_hit_rate=top_15_hit_rate,
            sim_top_1_hit_rate=sim_top_1_hit_rate,
            sim_top_3_hit_rate=sim_top_3_hit_rate,
            sim_top_5_hit_rate=sim_top_5_hit_rate,
            real_top_1_hit_rate=real_top_1_hit_rate,
            real_top_3_hit_rate=real_top_3_hit_rate,
            real_top_5_hit_rate=real_top_5_hit_rate,
            sim_average_hit_rate=sim_average_hit_rate,
            real_average_hit_rate=real_average_hit_rate,
            average_hit_rate=average_hit_rate,
            sim_total_scenarios=number_sim,
            real_total_scenarios=total - number_sim,
            total_scenarios=total,
            sim_top_1_hits=sim_hits[1],
            sim_top_3_hits=sim_hits[3],
            sim_top_5_hits=sim_hits[5],
            real_top_1_hits=real_hits[1],
            real_top_3_hits=real_hits[3],
            real_top_5_hits=real_hits[5],
            top_1_hits=hits[1],
            top_3_hits=hits[3],
            top_5_hits=hits[5]
        )

        self.save_metrics(metrics)
        return metrics

class SimulationEvaluator(BaseEvaluator):
    """Evaluator for simulation tasks"""
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = self._get_device(device)
        
        pipeline_device = self.device
        st_device = "cuda" if self.device == 0 else "cpu" 
        
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=5,
            device=pipeline_device
        )
        self.topic_model = SentenceTransformer(
            model_name_or_path=os.getenv("TOPIC_MODEL_PATH", "paraphrase-MiniLM-L6-v2"),
            device=st_device,
        )
        
    def _get_device(self, device: str) -> int:
        """Parse device from string"""
        if device == "gpu":
            if torch.cuda.is_available():
                return 0  # GPU
            else:
                logging.warning("GPU is not available, falling back to CPU")
                return -1  # CPU
        elif device == "cpu":
            return -1  # CPU
        elif device == "auto":
            return 0 if torch.cuda.is_available() else -1
        else:
            raise ValueError("Device type must be 'cpu', 'gpu' or 'auto'")

    def calculate_metrics(
        self,
        simulated_data: list[dict],
        real_data: list[dict],
        number_sim: int
    ) -> SimulationMetrics:
        """Calculate all simulation metrics"""
        # Calculate star error
        simulated_stars = [item['stars'] for item in simulated_data]
        real_stars = [item['stars'] for item in real_data]
        star_error = 0
        sim_star_error = 0
        real_star_error = 0
        index = 0
        for sim_star, real_star in zip(simulated_stars, real_stars):
            if sim_star > 5:
                sim_star = 5
            elif sim_star < 0:
                sim_star = 0
            star_error += abs(sim_star - real_star) / 5
            if index < number_sim:
                sim_star_error += abs(sim_star - real_star) / 5
            else:
                real_star_error += abs(sim_star - real_star) / 5
        star_error = star_error / len(real_stars)
        sim_star_error = sim_star_error / number_sim if number_sim > 0 else 0
        real_star_error = real_star_error / (len(real_stars) - number_sim) if len(real_stars) - number_sim > 0 else 0
        preference_estimation = 1 - star_error
        sim_preference_estimation = 1 - sim_star_error
        real_preference_estimation = 1 - real_star_error

        # Calculate review metrics
        simulated_reviews = [item['review'] for item in simulated_data]
        real_reviews = [item['review'] for item in real_data]
        review_details = self._calculate_review_metrics(
            simulated_reviews,
            real_reviews,
            number_sim
        )

        sentiment_error = review_details['sentiment_error']
        sim_sentiment_error = review_details['sim_sentiment_error']
        real_sentiment_error = review_details['real_sentiment_error']
        emotion_error = review_details['emotion_error']
        sim_emotion_error = review_details['sim_emotion_error']
        real_emotion_error = review_details['real_emotion_error']
        topic_error = review_details['topic_error']
        sim_topic_error = review_details['sim_topic_error']
        real_topic_error = review_details['real_topic_error']
        review_generation = 1 - (sentiment_error * 0.25 + emotion_error * 0.25 + topic_error * 0.5)
        sim_review_generation = 1 - (sim_sentiment_error * 0.25 + sim_emotion_error * 0.25 + sim_topic_error * 0.5)
        real_review_generation = 1 - (real_sentiment_error * 0.25 + real_emotion_error * 0.25 + real_topic_error * 0.5)
        overall_quality = (preference_estimation + review_generation) / 2
        sim_overall_quality = (sim_preference_estimation + sim_review_generation) / 2
        real_overall_quality = (real_preference_estimation + real_review_generation) / 2

        metrics = SimulationMetrics(
            sim_preference_estimation=sim_preference_estimation,
            real_preference_estimation=real_preference_estimation,
            preference_estimation=preference_estimation,
            sim_sentiment_error=sim_sentiment_error,
            real_sentiment_error=real_sentiment_error,
            sentiment_error=sentiment_error,
            sim_emotion_error=sim_emotion_error,
            real_emotion_error=real_emotion_error,
            emotion_error=emotion_error,
            sim_topic_error=sim_topic_error,
            real_topic_error=real_topic_error,
            topic_error=topic_error,
            sim_review_generation=sim_review_generation,
            real_review_generation=real_review_generation,
            review_generation=review_generation,
            sim_overall_quality=sim_overall_quality,
            real_overall_quality=real_overall_quality,
            overall_quality=overall_quality
        )

        fine_grained_metrics = {}
        for i, real_data, simulated_data in zip(range(len(real_data)), real_data, simulated_data):
            fine_grained_metrics[i] = {
                'gt_stars': real_data['stars'],
                'simulated_stars': simulated_data['stars'],
                'preference_estimation': 1 - abs(real_data['stars'] - simulated_data['stars']) / 5,
                'gt_review': real_data['review'],
                'simulated_review': simulated_data['review'],
                'sentiment_error': review_details['sentiment_error_list'][i],
                'emotion_error': review_details['emotion_error_list'][i],
                'topic_error': review_details['topic_error_list'][i],
                'is_simulated': i < number_sim
            }

        self.save_metrics(metrics)
        return metrics, fine_grained_metrics

    def _calculate_review_metrics(
        self,
        simulated_reviews: list[str],
        real_reviews: list[str],
        number_sim: int
    ) -> dict[str, float]:
        """Calculate detailed review metrics between two texts"""
        # sentiment analysis
        sentiment_error = []
        sim_sentiment_error = []
        real_sentiment_error = []
        emotion_error = []
        sim_emotion_error = []
        real_emotion_error = []
        topic_error = []
        sim_topic_error = []
        real_topic_error = []
        index = 0
        for simulated_review, real_review in zip(simulated_reviews, real_reviews):
            # sentiment analysis
            sentiment1 = self.sia.polarity_scores(simulated_review)['compound']
            sentiment2 = self.sia.polarity_scores(real_review)['compound']
            sentiment_error_single = abs(sentiment1 - sentiment2) / 2
            sentiment_error.append(sentiment_error_single)
            if index < number_sim:
                sim_sentiment_error.append(sentiment_error_single)
            else:
                real_sentiment_error.append(sentiment_error_single)

            # Topic analysis
            embeddings = self.topic_model.encode([simulated_review, real_review])
            topic_error_single = distance.cosine(embeddings[0], embeddings[1]) / 2
            topic_error.append(topic_error_single)
            if index < number_sim:
                sim_topic_error.append(topic_error_single)
            else:
                real_topic_error.append(topic_error_single)
            index += 1

        # Emotion analysis
        for i in range(len(simulated_reviews)):
            if len(simulated_reviews[i]) > 300:
                simulated_reviews[i] = simulated_reviews[i][:300]
            if len(real_reviews[i]) > 300:
                real_reviews[i] = real_reviews[i][:300]
        simulated_emotions = self.emotion_classifier(simulated_reviews)
        real_emotions = self.emotion_classifier(real_reviews)
        index = 0
        for sim_emotion, real_emotion in zip(simulated_emotions, real_emotions):
            emotion_error_single = self._calculate_emotion_error(sim_emotion, real_emotion)
            emotion_error.append(emotion_error_single)
            if index < number_sim:
                sim_emotion_error.append(emotion_error_single)
            else:
                real_emotion_error.append(emotion_error_single)
            index += 1

        sentiment_error_mean = np.mean(sentiment_error)
        sim_sentiment_error_mean = np.mean(sim_sentiment_error)
        real_sentiment_error_mean = np.mean(real_sentiment_error)
        emotion_error_mean = np.mean(emotion_error)
        sim_emotion_error_mean = np.mean(sim_emotion_error)
        real_emotion_error_mean = np.mean(real_emotion_error)
        topic_error_mean = np.mean(topic_error)
        sim_topic_error_mean = np.mean(sim_topic_error)
        real_topic_error_mean = np.mean(real_topic_error)
        return {
            'sentiment_error': sentiment_error_mean,
            'sim_sentiment_error': sim_sentiment_error_mean,
            'real_sentiment_error': real_sentiment_error_mean,
            'emotion_error': emotion_error_mean,
            'sim_emotion_error': sim_emotion_error_mean,
            'real_emotion_error': real_emotion_error_mean,
            'topic_error': topic_error_mean,
            'sim_topic_error': sim_topic_error_mean,
            'real_topic_error': real_topic_error_mean,
            'sentiment_error_list': sentiment_error,
            'emotion_error_list': emotion_error,
            'topic_error_list': topic_error
        }

    def _calculate_emotion_error(
        self,
        emotions1: list[dict],
        emotions2: list[dict]
    ) -> float:
        """Calculate similarity between two emotion distributions"""
        # Convert emotions to vectors
        emotion_dict1 = {e['label']: e['score'] for e in emotions1}
        emotion_dict2 = {e['label']: e['score'] for e in emotions2}

        # Get all unique emotions
        all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())

        # Create vectors
        vec1 = np.array([emotion_dict1.get(e, 0) for e in all_emotions])
        vec2 = np.array([emotion_dict2.get(e, 0) for e in all_emotions])

        # Calculate emotion error
        return float(np.mean(np.abs(vec1 - vec2)))
