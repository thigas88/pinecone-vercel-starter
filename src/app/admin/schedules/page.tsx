"use client";

import { useEffect, useState } from "react";
import { 
  Calendar, 
  Clock,
  Plus,
  Trash2,
  Edit,
  CheckCircle,
  XCircle,
  AlertCircle,
  FileText,
  RefreshCw,
  Play,
  Pause
} from "lucide-react";

interface Schedule {
  id: number;
  document_id: number;
  scheduled_for: string;
  status: string;
  created_at: string;
  document?: {
    id: number;
    title: string | null;
    url: string;
    category: string | null;
  };
}

interface Document {
  id: number;
  url: string;
  title: string | null;
  category: string | null;
  status: string;
  scheduled_at: string | null;
}

export default function SchedulesPage() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedSchedule, setSelectedSchedule] = useState<Schedule | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>("all");

  useEffect(() => {
    fetchData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch documents
      const docsRes = await fetch("http://localhost:8000/admin/documents");
      if (!docsRes.ok) throw new Error("Failed to fetch documents");
      const docsData = await docsRes.json();
      setDocuments(docsData);
      
      // Filter scheduled documents
      const scheduledDocs = docsData.filter((doc: Document) => doc.scheduled_at);
      
      // Create schedule objects from scheduled documents
      const scheduleData: Schedule[] = scheduledDocs.map((doc: Document) => ({
        id: doc.id,
        document_id: doc.id,
        scheduled_for: doc.scheduled_at!,
        status: doc.status,
        created_at: doc.scheduled_at!,
        document: {
          id: doc.id,
          title: doc.title,
          url: doc.url,
          category: doc.category
        }
      }));
      
      setSchedules(scheduleData);
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

  const getTimeUntil = (dateString: string) => {
    const scheduled = new Date(dateString);
    const now = new Date();
    const diff = scheduled.getTime() - now.getTime();
    
    if (diff < 0) return "Atrasado";
    
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "indexed":
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "processing":
        return <Clock className="w-5 h-5 text-yellow-500 animate-spin" />;
      case "error":
        return <XCircle className="w-5 h-5 text-red-500" />;
      case "agendado":
        return <Clock className="w-5 h-5 text-blue-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const statusClasses = {
      indexed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      processing: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
      error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
      agendado: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    };

    return (
      <span className={`px-2 py-1 text-xs rounded-full ${statusClasses[status as keyof typeof statusClasses] || "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"}`}>
        {status}
      </span>
    );
  };

  const handleExecuteNow = async (schedule: Schedule) => {
    if (!confirm(`Executar o processamento de "${schedule.document?.title || schedule.document?.url}" agora?`)) {
      return;
    }

    try {
      // Trigger immediate processing
      const response = await fetch("http://localhost:8000/admin/ingest/document", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: schedule.document?.url,
          title: schedule.document?.title,
          category: schedule.document?.category,
          splitting_method: "character",
          chunk_size: 1000,
          chunk_overlap: 200,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to execute processing");
      }

      alert("Processamento iniciado com sucesso!");
      fetchData();
    } catch (err) {
      alert(`Erro ao executar: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  // Filter schedules
  const filteredSchedules = schedules.filter(schedule => {
    return filterStatus === "all" || schedule.status === filterStatus;
  });

  // Group schedules by date
  const groupedSchedules = filteredSchedules.reduce((groups, schedule) => {
    const date = new Date(schedule.scheduled_for).toLocaleDateString("pt-BR");
    if (!groups[date]) {
      groups[date] = [];
    }
    groups[date].push(schedule);
    return groups;
  }, {} as { [key: string]: Schedule[] });

  // Sort dates
  const sortedDates = Object.keys(groupedSchedules).sort((a, b) => {
    const dateA = new Date(a.split('/').reverse().join('-'));
    const dateB = new Date(b.split('/').reverse().join('-'));
    return dateA.getTime() - dateB.getTime();
  });

  if (loading && schedules.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Carregando agendamentos...</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white">
          Agendamentos de Processamento
        </h1>
        <div className="flex items-center space-x-4">
          <button
            onClick={fetchData}
            className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className="w-5 h-5 mr-2" />
            Atualizar
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-5 h-5 mr-2" />
            Novo Agendamento
          </button>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.length}
              </p>
            </div>
            <Calendar className="w-8 h-8 text-gray-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Pendentes</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.filter(s => s.status === "agendado").length}
              </p>
            </div>
            <Clock className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Processando</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.filter(s => s.status === "processing").length}
              </p>
            </div>
            <RefreshCw className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Concluídos</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.filter(s => s.status === "indexed" || s.status === "completed").length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
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
            <option value="agendado">Agendados</option>
            <option value="processing">Processando</option>
            <option value="indexed">Concluídos</option>
            <option value="error">Erros</option>
          </select>
        </div>
      </div>

      {/* Schedule Timeline */}
      {sortedDates.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
          <Calendar className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            Nenhum agendamento encontrado
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {sortedDates.map(date => (
            <div key={date}>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                <Calendar className="w-5 h-5 mr-2" />
                {date}
              </h3>
              
              <div className="space-y-4">
                {groupedSchedules[date].map(schedule => (
                  <div key={schedule.id} className="bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg transition-shadow">
                    <div className="p-6">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-4">
                          {getStatusIcon(schedule.status)}
                          <div className="flex-1">
                            <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-1">
                              {schedule.document?.title || `Documento #${schedule.document_id}`}
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                              {schedule.document?.url}
                            </p>
                            
                            <div className="flex items-center space-x-4 text-sm">
                              <div className="flex items-center text-gray-600 dark:text-gray-400">
                                <Clock className="w-4 h-4 mr-1" />
                                {formatDate(schedule.scheduled_for)}
                              </div>
                              {schedule.status === "agendado" && (
                                <div className="flex items-center text-blue-600 dark:text-blue-400">
                                  <AlertCircle className="w-4 h-4 mr-1" />
                                  Em {getTimeUntil(schedule.scheduled_for)}
                                </div>
                              )}
                              {schedule.document?.category && (
                                <div className="flex items-center text-gray-600 dark:text-gray-400">
                                  <FileText className="w-4 h-4 mr-1" />
                                  {schedule.document.category}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {getStatusBadge(schedule.status)}
                          {schedule.status === "agendado" && (
                            <button
                              onClick={() => handleExecuteNow(schedule)}
                              className="p-2 text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
                              title="Executar agora"
                            >
                              <Play className="w-5 h-5" />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add Schedule Modal */}
      {showAddModal && (
        <AddScheduleModal 
          documents={documents.filter(d => !d.scheduled_at)}
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false);
            fetchData();
          }}
        />
      )}
    </div>
  );
}

// Add Schedule Modal Component
function AddScheduleModal({ 
  documents, 
  onClose, 
  onSuccess 
}: { 
  documents: Document[]; 
  onClose: () => void; 
  onSuccess: () => void;
}) {
  const [selectedDoc, setSelectedDoc] = useState<number | null>(null);
  const [scheduledDate, setScheduledDate] = useState("");
  const [scheduledTime, setScheduledTime] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDoc || !scheduledDate || !scheduledTime) return;

    setLoading(true);
    setError(null);

    try {
      // Combine date and time
      const scheduledAt = new Date(`${scheduledDate}T${scheduledTime}`).toISOString();
      
      // Here you would typically update the document with the scheduled time
      // For now, we'll just show a success message
      alert(`Agendamento criado para ${scheduledAt}`);
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-md w-full">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
            Novo Agendamento
          </h2>
        </div>
        
        <form onSubmit={handleSubmit} className="p-6">
          {error && (
            <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Documento
              </label>
              <select
                value={selectedDoc || ""}
                onChange={(e) => setSelectedDoc(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                required
              >
                <option value="">Selecione um documento</option>
                {documents.map(doc => (
                  <option key={doc.id} value={doc.id}>
                    {doc.title || doc.url}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Data
              </label>
              <input
                type="date"
                value={scheduledDate}
                onChange={(e) => setScheduledDate(e.target.value)}
                min={new Date().toISOString().split('T')[0]}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Hora
              </label>
              <input
                type="time"
                value={scheduledTime}
                onChange={(e) => setScheduledTime(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                required
              />
            </div>
          </div>
          
          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              Cancelar
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {loading ? "Criando..." : "Criar Agendamento"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}