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
    active_scenario_id = db.Column(db.Integer, db.ForeignKey('scenarios.id', use_alter=True), nullable=True)
    
    # Relationships
    cycles = db.relationship('Cycle', backref='project', lazy=True, cascade='all, delete-orphan')
    scenarios = db.relationship('Scenario', backref='project', lazy=True, cascade='all, delete-orphan',
                                foreign_keys='Scenario.project_id')
    
    def to_dict(self, include_scenarios=False):
        data = {
            'id': self.id,
            'name': self.name,
            'dataset_path': self.dataset_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'cycle_count': len(self.cycles),
            'active_scenario_id': self.active_scenario_id
        }
        if include_scenarios:
            data['scenarios'] = [s.to_dict() for s in self.scenarios]
        return data
    
    def get_progress(self):
        """Calculate current progress against active scenario."""
        if not self.active_scenario_id:
            return None
        
        active_scenario = Scenario.query.get(self.active_scenario_id)
        if not active_scenario:
            return None
        
        # Calculate progress metrics
        total_samples_tested = sum(len(c.samples) for c in self.cycles)
        completed_samples = sum(
            sum(1 for s in c.samples if s.status == 'completed') 
            for c in self.cycles
        )
        
        planned_total = active_scenario.initial_samples + (
            active_scenario.planned_cycles * active_scenario.samples_per_cycle
        )
        
        return {
            'cycles_completed': len(self.cycles),
            'cycles_planned': active_scenario.planned_cycles,
            'samples_tested': completed_samples,
            'samples_planned': planned_total,
            'cost_spent': completed_samples * active_scenario.cost_per_sample,
            'cost_budget': planned_total * active_scenario.cost_per_sample,
            'days_elapsed': (datetime.utcnow() - self.created_at).days if self.created_at else 0,
            'days_planned': active_scenario.duration_per_cycle_days * active_scenario.planned_cycles,
            'coverage_current': (completed_samples / planned_total * 100) if planned_total > 0 else 0,
            'coverage_target': active_scenario.target_coverage
        }


class Scenario(db.Model):
    """A scenario represents an experimental plan for a project."""
    __tablename__ = 'scenarios'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    
    # Planning parameters
    planned_cycles = db.Column(db.Integer, default=2)
    samples_per_cycle = db.Column(db.Integer, default=5)
    initial_samples = db.Column(db.Integer, default=10)
    duration_per_cycle_days = db.Column(db.Integer, default=30)
    cost_per_sample = db.Column(db.Float, default=100.0)
    target_coverage = db.Column(db.Float, default=10.0)  # Target coverage percentage
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    
    def to_dict(self):
        # Calculate estimated totals
        total_samples = self.initial_samples + (self.planned_cycles * self.samples_per_cycle)
        total_cost = total_samples * self.cost_per_sample
        total_duration = self.planned_cycles * self.duration_per_cycle_days
        
        return {
            'id': self.id,
            'project_id': self.project_id,
            'name': self.name,
            'planned_cycles': self.planned_cycles,
            'samples_per_cycle': self.samples_per_cycle,
            'initial_samples': self.initial_samples,
            'duration_per_cycle_days': self.duration_per_cycle_days,
            'cost_per_sample': self.cost_per_sample,
            'target_coverage': self.target_coverage,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'notes': self.notes,
            # Computed fields
            'total_samples': total_samples,
            'total_cost': total_cost,
            'total_duration_days': total_duration
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
