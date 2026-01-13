"""
Database Models for Results Page

SQLite database models using Flask-SQLAlchemy for tracking:
- Projects: Named containers linked to datasets
- Cycles: Batches of samples sent to the lab
- Samples: Individual samples with predictions and lab results
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class Project(db.Model):
    """A project represents a material discovery campaign linked to a dataset."""
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    dataset_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    cycles = db.relationship('Cycle', backref='project', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'dataset_path': self.dataset_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'cycle_count': len(self.cycles)
        }


class Cycle(db.Model):
    """A cycle represents a batch of samples sent to the lab."""
    __tablename__ = 'cycles'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    cycle_number = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    lab_result_columns = db.Column(db.Text, nullable=True)  # JSON array of column names for lab results
    
    # Relationship
    samples = db.relationship('Sample', backref='cycle', lazy=True, cascade='all, delete-orphan')
    
    def get_lab_result_columns(self):
        return json.loads(self.lab_result_columns) if self.lab_result_columns else []
    
    def set_lab_result_columns(self, columns):
        self.lab_result_columns = json.dumps(columns)
    
    def to_dict(self, include_samples=False):
        data = {
            'id': self.id,
            'project_id': self.project_id,
            'cycle_number': self.cycle_number,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'notes': self.notes,
            'lab_result_columns': self.get_lab_result_columns(),
            'sample_count': len(self.samples),
            'pending_count': sum(1 for s in self.samples if s.status == 'pending'),
            'completed_count': sum(1 for s in self.samples if s.status == 'completed')
        }
        if include_samples:
            data['samples'] = [s.to_dict() for s in self.samples]
        return data


class Sample(db.Model):
    """A sample represents a single experiment with predictions and lab results."""
    __tablename__ = 'samples'
    
    id = db.Column(db.Integer, primary_key=True)
    cycle_id = db.Column(db.Integer, db.ForeignKey('cycles.id'), nullable=False)
    idx_sample = db.Column(db.Integer, nullable=False)  # IDX_SAMPLE from dataset
    
    # JSON fields for flexibility
    row_data = db.Column(db.Text, nullable=True)  # Original row data as JSON
    predictions = db.Column(db.Text, nullable=True)  # Model predictions as JSON
    lab_results = db.Column(db.Text, nullable=True)  # Lab results as JSON
    
    # Status tracking
    status = db.Column(db.String(20), default='pending')  # pending, tested, completed
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_row_data(self):
        return json.loads(self.row_data) if self.row_data else {}
    
    def set_row_data(self, data):
        self.row_data = json.dumps(data)
    
    def get_predictions(self):
        return json.loads(self.predictions) if self.predictions else {}
    
    def set_predictions(self, data):
        self.predictions = json.dumps(data)
    
    def get_lab_results(self):
        return json.loads(self.lab_results) if self.lab_results else {}
    
    def set_lab_results(self, data):
        self.lab_results = json.dumps(data)
        # Auto-update status based on whether results are provided
        if data and any(v is not None and v != '' for v in data.values()):
            self.status = 'completed'
    
    def to_dict(self):
        return {
            'id': self.id,
            'cycle_id': self.cycle_id,
            'idx_sample': self.idx_sample,
            'row_data': self.get_row_data(),
            'predictions': self.get_predictions(),
            'lab_results': self.get_lab_results(),
            'status': self.status,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
