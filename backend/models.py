from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    url = Column(Text, nullable=False)
    title = Column(Text)
    tags = Column(Text)
    category = Column(Text)
    splitting_method = Column(Text)
    chunk_size = Column(Integer)
    overlap = Column(Integer)
    file_name = Column(Text)
    status = Column(String, default='agendado')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    scheduled_at = Column(DateTime(timezone=True))
    last_indexed_at = Column(DateTime(timezone=True))
    history = relationship('IngestHistory', back_populates='document', cascade='all, delete-orphan')
    schedules = relationship('IngestSchedule', back_populates='document', cascade='all, delete-orphan')

class IngestHistory(Base):
    __tablename__ = 'ingest_history'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True))
    status = Column(String)
    message = Column(Text)
    chunks_indexed = Column(Integer)
    error = Column(Text)
    document = relationship('Document', back_populates='history')

class IngestSchedule(Base):
    __tablename__ = 'ingest_schedule'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    scheduled_for = Column(DateTime(timezone=True), nullable=False)
    status = Column(String, default='agendado')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    document = relationship('Document', back_populates='schedules')

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    ingest_history_id = Column(Integer, ForeignKey('ingest_history.id', ondelete='SET NULL'))
    meta = Column(JSONB, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    hash = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship('Document', backref='chunks')
    ingest_history = relationship('IngestHistory', backref='chunks')