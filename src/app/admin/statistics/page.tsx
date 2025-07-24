"use client";

import { useEffect, useState } from "react";
import { 
  BarChart3, 
  TrendingUp,
  Database,
  FileText,
  Clock,
  Calendar,
  PieChart,
  Activity,
  Hash,
  Layers
} from "lucide-react";

interface Stats {
  total_chunks: number;
  total_documents: number;
  oldest_chunk: string;
  newest_chunk: string;
  table_size: string;
  category_distribution: Array<{
    category: string;
    count: number;
  }>;
}

interface Document {
  id: number;
  title: string | null;
  status: string;
  created_at: string;
  category: string | null;
  splitting_method: string | null;
  chunk_size: number | null;
}

interface ProcessingStats {
  totalProcessings: number;
  successRate: number;
  averageProcessingTime: number;
  processingsByDay: { [key: string]: number };
  errorsByType: { [key: string]: number };
}

export default function StatisticsPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [processingStats, setProcessingStats] = useState<ProcessingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState("7d");

  useEffect(() => {
    fetchAllData();
  }, [timeRange]);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      
      // Fetch vector store stats
      const statsRes = await fetch("http://localhost:8000/admin/vector-store/stats");
      if (!statsRes.ok) throw new Error("Failed to fetch stats");
      const statsData = await statsRes.json();
      setStats(statsData);
      
      // Fetch documents
      const docsRes = await fetch("http://localhost:8000/admin/documents");
      if (!docsRes.ok) throw new Error("Failed to fetch documents");
      const docsData = await docsRes.json();
      setDocuments(docsData);
      
      // Calculate processing statistics
      await calculateProcessingStats(docsData);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const calculateProcessingStats = async (docs: Document[]) => {
    try {
      let totalProcessings = 0;
      let successfulProcessings = 0;
      let totalProcessingTime = 0;
      const processingsByDay: { [key: string]: number } = {};
      const errorsByType: { [key: string]: number } = {};
      
      // Fetch processing history for each document
      for (const doc of docs) {
        const historyRes = await fetch(`http://localhost:8000/admin/documents/${doc.id}/history`);
        if (historyRes.ok) {
          const historyData = await historyRes.json();
          
          historyData.forEach((history: any) => {
            totalProcessings++;
            
            if (history.status === "completed") {
              successfulProcessings++;
            } else if (history.status === "error" && history.error) {
              const errorType = history.error.split(':')[0] || "Unknown";
              errorsByType[errorType] = (errorsByType[errorType] || 0) + 1;
            }
            
            // Calculate processing time
            if (history.finished_at) {
              const duration = new Date(history.finished_at).getTime() - new Date(history.started_at).getTime();
              totalProcessingTime += duration;
            }
            
            // Count by day
            const day = new Date(history.started_at).toLocaleDateString("pt-BR");
            processingsByDay[day] = (processingsByDay[day] || 0) + 1;
          });
        }
      }
      
      setProcessingStats({
        totalProcessings,
        successRate: totalProcessings > 0 ? (successfulProcessings / totalProcessings) * 100 : 0,
        averageProcessingTime: totalProcessings > 0 ? totalProcessingTime / totalProcessings : 0,
        processingsByDay,
        errorsByType
      });
    } catch (err) {
      console.error("Error calculating processing stats:", err);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleString("pt-BR");
  };

  const formatDuration = (milliseconds: number) => {
    const seconds = Math.floor(milliseconds / 1000);
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

  // Calculate additional statistics
  const documentsByStatus = documents.reduce((acc, doc) => {
    acc[doc.status] = (acc[doc.status] || 0) + 1;
    return acc;
  }, {} as { [key: string]: number });

  const documentsBySplittingMethod = documents.reduce((acc, doc) => {
    const method = doc.splitting_method || "character";
    acc[method] = (acc[method] || 0) + 1;
    return acc;
  }, {} as { [key: string]: number });

  const averageChunkSize = documents.reduce((sum, doc) => sum + (doc.chunk_size || 1000), 0) / (documents.length || 1);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Carregando estatísticas...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-800 dark:text-red-200">Erro: {error}</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white">
          Estatísticas do Sistema
        </h1>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
        >
          <option value="7d">Últimos 7 dias</option>
          <option value="30d">Últimos 30 dias</option>
          <option value="90d">Últimos 90 dias</option>
          <option value="all">Todo período</option>
        </select>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <FileText className="w-8 h-8 text-blue-500" />
            <span className="text-sm text-gray-500 dark:text-gray-400">Total</span>
          </div>
          <p className="text-3xl font-bold text-gray-800 dark:text-white">
            {stats?.total_documents || 0}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Documentos
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <Database className="w-8 h-8 text-green-500" />
            <span className="text-sm text-gray-500 dark:text-gray-400">Total</span>
          </div>
          <p className="text-3xl font-bold text-gray-800 dark:text-white">
            {stats?.total_chunks || 0}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Chunks
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="w-8 h-8 text-purple-500" />
            <span className="text-sm text-gray-500 dark:text-gray-400">Taxa</span>
          </div>
          <p className="text-3xl font-bold text-gray-800 dark:text-white">
            {processingStats?.successRate.toFixed(1) || 0}%
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Sucesso
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <Layers className="w-8 h-8 text-orange-500" />
            <span className="text-sm text-gray-500 dark:text-gray-400">Tamanho</span>
          </div>
          <p className="text-3xl font-bold text-gray-800 dark:text-white">
            {stats?.table_size || "0 MB"}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Banco de Dados
          </p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Category Distribution */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white flex items-center">
              <PieChart className="w-5 h-5 mr-2" />
              Distribuição por Categoria
            </h2>
          </div>
          <div className="p-6">
            {!stats?.category_distribution || stats.category_distribution.length === 0 ? (
              <p className="text-gray-500 dark:text-gray-400">Nenhuma categoria encontrada</p>
            ) : (
              <div className="space-y-4">
                {stats.category_distribution.map((cat) => {
                  const percentage = ((cat.count / (stats.total_chunks || 1)) * 100).toFixed(1);
                  return (
                    <div key={cat.category}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          {cat.category}
                        </span>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {cat.count} chunks ({percentage}%)
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Document Status Distribution */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Status dos Documentos
            </h2>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {Object.entries(documentsByStatus).map(([status, count]) => {
                const percentage = ((count / documents.length) * 100).toFixed(1);
                const colors = {
                  indexed: "bg-green-600",
                  processing: "bg-yellow-600",
                  error: "bg-red-600",
                  agendado: "bg-blue-600"
                };
                return (
                  <div key={status}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                        {status}
                      </span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {count} ({percentage}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className={`${colors[status as keyof typeof colors] || "bg-gray-600"} h-2 rounded-full transition-all duration-500`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Processing Methods */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white flex items-center">
              <Hash className="w-5 h-5 mr-2" />
              Métodos de Divisão
            </h2>
          </div>
          <div className="p-6">
            <div className="space-y-3">
              {Object.entries(documentsBySplittingMethod).map(([method, count]) => (
                <div key={method} className="flex justify-between items-center">
                  <span className="text-sm text-gray-700 dark:text-gray-300 capitalize">
                    {method}
                  </span>
                  <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-sm">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Processing Times */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white flex items-center">
              <Clock className="w-5 h-5 mr-2" />
              Tempos de Processamento
            </h2>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Tempo Médio</p>
                <p className="text-2xl font-bold text-gray-800 dark:text-white">
                  {formatDuration(processingStats?.averageProcessingTime || 0)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total de Processamentos</p>
                <p className="text-xl font-semibold text-gray-800 dark:text-white">
                  {processingStats?.totalProcessings || 0}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Chunk Size Médio</p>
                <p className="text-xl font-semibold text-gray-800 dark:text-white">
                  {Math.round(averageChunkSize)} caracteres
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white flex items-center">
              <Calendar className="w-5 h-5 mr-2" />
              Informações do Sistema
            </h2>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Chunk Mais Antigo</p>
                <p className="text-sm font-medium text-gray-800 dark:text-white">
                  {formatDate(stats?.oldest_chunk || null)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Chunk Mais Recente</p>
                <p className="text-sm font-medium text-gray-800 dark:text-white">
                  {formatDate(stats?.newest_chunk || null)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Categorias Únicas</p>
                <p className="text-xl font-semibold text-gray-800 dark:text-white">
                  {stats?.category_distribution?.length || 0}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}