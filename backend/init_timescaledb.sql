-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vectorscale extension for vector operations
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Tabela principal de documentos indexados
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    tags TEXT,
    category TEXT,
    splitting_method TEXT,
    chunk_size INTEGER,
    overlap INTEGER,
    file_name TEXT,
    status TEXT NOT NULL DEFAULT 'agendado',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_at TIMESTAMPTZ,
    last_indexed_at TIMESTAMPTZ
);

-- Histórico de execuções de ingestão
CREATE TABLE IF NOT EXISTS ingest_history (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    message TEXT,
    chunks_indexed INTEGER,
    error TEXT
);

-- Agendamentos futuros de ingestão
CREATE TABLE IF NOT EXISTS ingest_schedule (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    scheduled_for TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'agendado',
    recurrence_type TEXT, -- 'daily', 'weekly', 'monthly', 'custom'
    recurrence_interval INTEGER, -- interval in days for custom recurrence
    recurrence_days_of_week JSONB, -- for weekly: ['monday', 'tuesday', etc]
    recurrence_day_of_month INTEGER, -- for monthly: 1-31
    recurrence_time TIME, -- time of day for recurrence
    next_execution TIMESTAMPTZ,
    last_execution TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabela de categorias
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    color TEXT, -- hex color for UI
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default categories
INSERT INTO categories (name, description, color) VALUES
    ('assinatura', 'Documentos relacionados a assinaturas digitais', '#3B82F6'),
    ('ecampus', 'Sistema e-Campus', '#10B981'),
    ('pag', 'Pagamentos e financeiro', '#F59E0B'),
    ('sei', 'Sistema SEI', '#8B5CF6'),
    ('revista', 'Revistas e publicações', '#EC4899'),
    ('wifi', 'Rede e conectividade', '#06B6D4'),
    ('mautic', 'Sistema Mautic', '#6366F1'),
    ('metabase', 'Analytics e BI', '#84CC16'),
    ('evoto', 'Sistema de votação', '#F97316'),
    ('outros', 'Outros documentos', '#6B7280')
ON CONFLICT (name) DO NOTHING;

-- Tabela de chunks/textos extraídos
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
	metadata JSONB,
    ingest_history_id INTEGER REFERENCES ingest_history(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create vector store table for LangChain TimescaleVector
CREATE TABLE IF NOT EXISTS document_chunks_vector (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    contents TEXT NOT NULL,
    embedding vector(1536), -- Dimension for all-MiniLM-L6-v2 model  default 1536
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
	PRIMARY KEY (id, created_at)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable(
    'document_chunks_vector', 
    'created_at',
    if_not_exists => TRUE
);

-- Índice para busca rápida por status/data
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_ingest_schedule_status ON ingest_schedule(status);
CREATE INDEX IF NOT EXISTS idx_ingest_schedule_next_execution ON ingest_schedule(next_execution) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_ingest_schedule_document_id ON ingest_schedule(document_id);
CREATE INDEX IF NOT EXISTS idx_ingest_history_document_id ON ingest_history(document_id);
CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);

-- Indexes for document chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_ingest_history_id ON document_chunks(ingest_history_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_hash ON document_chunks(hash);

-- Vector indexes for similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_embedding 
ON document_chunks_vector 
USING diskann (embedding vector_cosine_ops); -- cosine distance

CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_document_id 
ON document_chunks_vector (document_id, created_at);

CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_metadata 
ON document_chunks_vector 
USING gin (metadata);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE
    ON documents FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_document_chunks_vector_updated_at BEFORE UPDATE
    ON document_chunks_vector FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_ingest_schedule_updated_at BEFORE UPDATE
    ON ingest_schedule FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_categories_updated_at BEFORE UPDATE
    ON categories FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();