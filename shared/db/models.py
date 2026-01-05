from sqlalchemy import Column, Integer, String, Float, DateTime, func, JSON, Boolean
from shared.config.database import Base  # remove "app." if you import relative to train_models.py
import datetime

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    confusion_matrix = Column(JSON)
    version = Column(String(20))
    version_hash = Column(String(20))
    is_production = Column(Boolean, default=False)
    feature_count = Column(Integer)
    sample_count = Column(Integer)
    training_time = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FraudTransaction(Base):
    __tablename__ = "fraud_transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    txn_id = Column(String(50), nullable=False, unique=True)
    country = Column(String(50))
    city = Column(String(50))
    merchant = Column(String(100))
    card_last4 = Column(Integer)
    is_fraud = Column(Integer)  # 1 for fraud, 0 for legitimate
    amount = Column(String)
    score = Column(Float)       # model confidence score
    created_at = Column(DateTime(timezone=True), server_default=func.now())
