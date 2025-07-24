"use client";

import { useEffect, useState } from "react";
import { 
  Activity, 
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  FileText,
  Calendar,
  Hash,
  MessageSquare
} from "lucide-react";

interface IngestHistory {
  id: number;
  document_id: number;
  started_at: string;
  finished_at: string | null;
  status: string;
  message: string | null;
  chunks_indexed: number | null;
  error: string | null;
}

interface Document {
  id: number;
  url: string;
  title: string | null;
  category: string | null;
  status: string;
}

interface ProcessingWithDocument extends IngestHistory {
  document?: Document;
}

export default function ProcessingPage() {
  const [processings, setProcessings] = useState<ProcessingWithDocument[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [refreshInterval, setRefreshInterval] = useState<number | null>(5000);

  useEffect(() => {
    fetchData();
    
    // Auto-refresh for active processings
    if (refreshInterval) {
      const interval = setInterval(() => {
        const hasActiveProcessing = processings.some(p => p.status === "processing");
        if (hasActiveProcessing) {
          fetchData();
        }
      }, refreshInterval);
      
      return () => clearInterval(interval);
    }
  }, [refreshInterval, processings.length]);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch all documents
      const docsRes = await fetch("http://localhost:8000/admin/documents");
      if (!docsRes.ok) throw new Error("Failed to fetch documents");
      const docsData = await docsRes.json();
      setDocuments(docsData);
      
      // Fetch all processing history
      const allProcessings: ProcessingWithDocument[] = [];
      
      for (const doc of docsData) {
        const historyRes = await fetch(`http://localhost:8000/admin/documents/${doc.id}/history`);
        if (historyRes.ok) {
          const historyData = await historyRes.json();
          historyData.forEach((history: IngestHistory) => {
            allProcessings.push({
              ...history,
              document: doc
            });
          });
        }
      }
      
      // Sort by started_at descending
      allProcessings.sort((a, b) => 
        new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
      );
      
      setProcessings(allProcessings);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleString("pt-BR");
  };

  const formatDuration = (start: string, end: string | null) => {
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const duration = endDate.getTime() - startDate.getTime();
    
    const seconds = Math.floor(duration / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "processing":
        return <Clock className="w-5 h-5 text-yellow-500 animate-spin" />;
      case "error":
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const statusClasses = {
      completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      processing: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
      error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    };

    return (
      <span className={`px-2 py-1 text-xs rounded-full ${statusClasses[status as keyof typeof statusClasses] || "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"}`}>
        {status}
      </span>
    );
  };

  // Filter processings
  const filteredProcessings = processings.filter(proc => {
    return filterStatus === "all" || proc.status === filterStatus;
  });

  // Calculate statistics
  const stats = {
    total: processings.length,
    completed: processings.filter(p => p.status === "completed").length,
    processing: processings.filter(p => p.status === "processing").length,
    error: processings.filter(p => p.status === "error").length,
    totalChunks: processings.reduce((sum, p) => sum + (p.chunks_indexed || 0), 0)
  };

  if (loading && processings.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Carregando processamentos...</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white">
          Gerenciamento de Processamentos
        </h1>
        <div className="flex items-center space-x-4">
          <select
            value={refreshInterval?.toString() || "null"}
            onChange={(e) => setRefreshInterval(e.target.value === "null" ? null : parseInt(e.target.value))}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="null">Sem auto-refresh</option>
            <option value="5000">Atualizar a cada 5s</option>
            <option value="10000">Atualizar a cada 10s</option>
            <option value="30000">Atualizar a cada 30s</option>
          </select>
          <button
            onClick={fetchData}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="w-5 h-5 mr-2" />
            Atualizar
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">{stats.total}</p>
            </div>
            <Activity className="w-8 h-8 text-gray-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Concluídos</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">{stats.completed}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Processando</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">{stats.processing}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Erros</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">{stats.error}</p>
            </div>
            <XCircle className="w-8 h-8 text-red-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Chunks</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">{stats.totalChunks}</p>
            </div>
            <Hash className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Filter */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Filtrar por status:
          </label>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="all">Todos</option>
            <option value="completed">Concluídos</option>
            <option value="processing">Processando</option>
            <option value="error">Erros</option>
          </select>
        </div>
      </div>

      {/* Processing List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        {filteredProcessings.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            Nenhum processamento encontrado
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {filteredProcessings.map((proc) => (
              <div key={proc.id} className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    {getStatusIcon(proc.status)}
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                          {proc.document?.title || `Documento #${proc.document_id}`}
                        </h3>
                        {getStatusBadge(proc.status)}
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600 dark:text-gray-400">
                        <div className="space-y-1">
                          <div className="flex items-center">
                            <Calendar className="w-4 h-4 mr-2" />
                            <span>Iniciado: {formatDate(proc.started_at)}</span>
                          </div>
                          {proc.finished_at && (
                            <div className="flex items-center">
                              <CheckCircle className="w-4 h-4 mr-2" />
                              <span>Finalizado: {formatDate(proc.finished_at)}</span>
                            </div>
                          )}
                          <div className="flex items-center">
                            <Clock className="w-4 h-4 mr-2" />
                            <span>Duração: {formatDuration(proc.started_at, proc.finished_at)}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          {proc.chunks_indexed !== null && (
                            <div className="flex items-center">
                              <Hash className="w-4 h-4 mr-2" />
                              <span>Chunks indexados: {proc.chunks_indexed}</span>
                            </div>
                          )}
                          {proc.document?.category && (
                            <div className="flex items-center">
                              <FileText className="w-4 h-4 mr-2" />
                              <span>Categoria: {proc.document.category}</span>
                            </div>
                          )}
                          {proc.message && (
                            <div className="flex items-center">
                              <MessageSquare className="w-4 h-4 mr-2" />
                              <span>{proc.message}</span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {proc.error && (
                        <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                          <p className="text-sm text-red-800 dark:text-red-200">
                            <strong>Erro:</strong> {proc.error}
                          </p>
                        </div>
                      )}
                      
                      {proc.document?.url && (
                        <div className="mt-3">
                          <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                            URL: {proc.document.url}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}