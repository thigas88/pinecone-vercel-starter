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
  Pause,
  Repeat,
  CalendarDays,
  Timer
} from "lucide-react";

interface Schedule {
  id: number;
  document_id: number;
  scheduled_for: string;
  status: string;
  recurrence_type: string | null;
  recurrence_interval: number | null;
  recurrence_days_of_week: string[] | null;
  recurrence_day_of_month: number | null;
  recurrence_time: string | null;
  next_execution: string | null;
  last_execution: string | null;
  is_active: string;
  created_at: string;
  updated_at: string;
}

interface Document {
  id: number;
  url: string;
  title: string | null;
  category: string | null;
  status: string;
}

interface ScheduleForm {
  document_id: number | null;
  scheduled_for: string;
  scheduled_time: string;
  recurrence_type: string;
  recurrence_interval: number;
  recurrence_days_of_week: string[];
  recurrence_day_of_month: number;
  recurrence_time: string;
}

const DAYS_OF_WEEK = [
  { value: 'monday', label: 'Segunda' },
  { value: 'tuesday', label: 'Terça' },
  { value: 'wednesday', label: 'Quarta' },
  { value: 'thursday', label: 'Quinta' },
  { value: 'friday', label: 'Sexta' },
  { value: 'saturday', label: 'Sábado' },
  { value: 'sunday', label: 'Domingo' }
];

export default function SchedulesPage() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingSchedule, setEditingSchedule] = useState<Schedule | null>(null);
  const [formData, setFormData] = useState<ScheduleForm>({
    document_id: null,
    scheduled_for: new Date().toISOString().split('T')[0],
    scheduled_time: "09:00",
    recurrence_type: "none",
    recurrence_interval: 1,
    recurrence_days_of_week: [],
    recurrence_day_of_month: 1,
    recurrence_time: "09:00"
  });

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch schedules
      const schedulesRes = await fetch("http://localhost:8000/admin/schedules");
      if (!schedulesRes.ok) throw new Error("Failed to fetch schedules");
      const schedulesData = await schedulesRes.json();
      setSchedules(schedulesData);
      
      // Fetch documents (only web documents)
      const docsRes = await fetch("http://localhost:8000/admin/documents");
      if (!docsRes.ok) throw new Error("Failed to fetch documents");
      const docsData = await docsRes.json();
      const webDocs = docsData.filter((doc: Document) => 
        doc.url && (doc.url.startsWith('http://') || doc.url.startsWith('https://'))
      );
      setDocuments(webDocs);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.document_id) return;

    try {
      const scheduledDateTime = new Date(`${formData.scheduled_for}T${formData.scheduled_time}`);
      
      const payload = {
        document_id: formData.document_id,
        scheduled_for: scheduledDateTime.toISOString(),
        recurrence_type: formData.recurrence_type === 'none' ? null : formData.recurrence_type,
        recurrence_interval: formData.recurrence_type === 'custom' ? formData.recurrence_interval : null,
        recurrence_days_of_week: formData.recurrence_type === 'weekly' ? formData.recurrence_days_of_week : null,
        recurrence_day_of_month: formData.recurrence_type === 'monthly' ? formData.recurrence_day_of_month : null,
        recurrence_time: formData.recurrence_type !== 'none' ? formData.recurrence_time : null
      };

      const url = editingSchedule 
        ? `http://localhost:8000/admin/schedules/${editingSchedule.id}`
        : "http://localhost:8000/admin/schedules";
      
      const method = editingSchedule ? "PUT" : "POST";

      const response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to save schedule");
      }

      alert(`Agendamento ${editingSchedule ? 'atualizado' : 'criado'} com sucesso!`);
      resetForm();
      fetchData();
    } catch (err) {
      alert(`Erro ao salvar agendamento: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  const handleExecuteNow = async (schedule: Schedule) => {
    if (!confirm("Executar este agendamento agora?")) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/admin/schedules/${schedule.id}/execute`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Failed to execute schedule");
      }

      alert("Execução iniciada com sucesso!");
      fetchData();
    } catch (err) {
      alert(`Erro ao executar: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  const handleDelete = async (schedule: Schedule) => {
    if (!confirm("Tem certeza que deseja desativar este agendamento?")) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/admin/schedules/${schedule.id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete schedule");
      }

      alert("Agendamento desativado com sucesso!");
      fetchData();
    } catch (err) {
      alert(`Erro ao desativar: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  const handleEdit = (schedule: Schedule) => {
    const scheduledDate = new Date(schedule.scheduled_for);
    
    setEditingSchedule(schedule);
    setFormData({
      document_id: schedule.document_id,
      scheduled_for: scheduledDate.toISOString().split('T')[0],
      scheduled_time: scheduledDate.toTimeString().slice(0, 5),
      recurrence_type: schedule.recurrence_type || 'none',
      recurrence_interval: schedule.recurrence_interval || 1,
      recurrence_days_of_week: schedule.recurrence_days_of_week || [],
      recurrence_day_of_month: schedule.recurrence_day_of_month || 1,
      recurrence_time: schedule.recurrence_time || "09:00"
    });
    setShowAddModal(true);
  };

  const resetForm = () => {
    setFormData({
      document_id: null,
      scheduled_for: new Date().toISOString().split('T')[0],
      scheduled_time: "09:00",
      recurrence_type: "none",
      recurrence_interval: 1,
      recurrence_days_of_week: [],
      recurrence_day_of_month: 1,
      recurrence_time: "09:00"
    });
    setEditingSchedule(null);
    setShowAddModal(false);
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleString("pt-BR");
  };

  const getRecurrenceText = (schedule: Schedule) => {
    if (!schedule.recurrence_type) return "Execução única";
    
    switch (schedule.recurrence_type) {
      case 'daily':
        return `Diariamente às ${schedule.recurrence_time}`;
      case 'weekly':
        const days = schedule.recurrence_days_of_week?.map(d => 
          DAYS_OF_WEEK.find(dw => dw.value === d)?.label
        ).join(', ');
        return `Semanalmente (${days}) às ${schedule.recurrence_time}`;
      case 'monthly':
        return `Mensalmente no dia ${schedule.recurrence_day_of_month} às ${schedule.recurrence_time}`;
      case 'custom':
        return `A cada ${schedule.recurrence_interval} dias às ${schedule.recurrence_time}`;
      default:
        return schedule.recurrence_type;
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
      case "agendado":
        return <Clock className="w-5 h-5 text-blue-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getDocumentById = (docId: number) => {
    return documents.find(d => d.id === docId);
  };

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
              <p className="text-sm text-gray-600 dark:text-gray-400">Ativos</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.filter(s => s.is_active === 'true').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Recorrentes</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.filter(s => s.recurrence_type !== null).length}
              </p>
            </div>
            <Repeat className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Únicos</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">
                {schedules.filter(s => s.recurrence_type === null).length}
              </p>
            </div>
            <Timer className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Schedules List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        {schedules.length === 0 ? (
          <div className="p-8 text-center">
            <Calendar className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 dark:text-gray-400">
              Nenhum agendamento encontrado
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Documento
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Próxima Execução
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Recorrência
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Última Execução
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Ações
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {schedules.map((schedule) => {
                  const document = getDocumentById(schedule.document_id);
                  return (
                    <tr key={schedule.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FileText className="w-5 h-5 text-gray-400 mr-3" />
                          <div>
                            <div className="text-sm font-medium text-gray-900 dark:text-white">
                              {document?.title || `Documento #${schedule.document_id}`}
                            </div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {document?.category || "Sem categoria"}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900 dark:text-white">
                          {formatDate(schedule.next_execution || schedule.scheduled_for)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center text-sm text-gray-900 dark:text-white">
                          {schedule.recurrence_type && <Repeat className="w-4 h-4 mr-2" />}
                          {getRecurrenceText(schedule)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {formatDate(schedule.last_execution)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {getStatusIcon(schedule.status)}
                          <span className="ml-2 text-sm text-gray-900 dark:text-white">
                            {schedule.is_active ? 'Ativo' : 'Inativo'}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={() => handleEdit(schedule)}
                            className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                            title="Editar"
                          >
                            <Edit className="w-5 h-5" />
                          </button>
                          <button
                            onClick={() => handleExecuteNow(schedule)}
                            className="text-green-600 hover:text-green-900 dark:text-green-400 dark:hover:text-green-300"
                            title="Executar agora"
                          >
                            <Play className="w-5 h-5" />
                          </button>
                          <button
                            onClick={() => handleDelete(schedule)}
                            className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                            title="Desativar"
                          >
                            <Trash2 className="w-5 h-5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Add/Edit Schedule Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
                {editingSchedule ? "Editar Agendamento" : "Novo Agendamento"}
              </h2>
            </div>
            
            <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[70vh]">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Documento (apenas documentos web)
                  </label>
                  <select
                    value={formData.document_id || ""}
                    onChange={(e) => setFormData({ ...formData, document_id: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    required
                  >
                    <option value="">Selecione um documento</option>
                    {documents.map(doc => (
                      <option key={doc.id} value={doc.id}>
                        {doc.title || doc.url} {doc.category && `(${doc.category})`}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Data Inicial
                    </label>
                    <input
                      type="date"
                      value={formData.scheduled_for}
                      onChange={(e) => setFormData({ ...formData, scheduled_for: e.target.value })}
                      min={new Date().toISOString().split('T')[0]}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Hora Inicial
                    </label>
                    <input
                      type="time"
                      value={formData.scheduled_time}
                      onChange={(e) => setFormData({ ...formData, scheduled_time: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                      required
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Tipo de Recorrência
                  </label>
                  <select
                    value={formData.recurrence_type}
                    onChange={(e) => setFormData({ ...formData, recurrence_type: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="none">Execução Única</option>
                    <option value="daily">Diariamente</option>
                    <option value="weekly">Semanalmente</option>
                    <option value="monthly">Mensalmente</option>
                    <option value="custom">Personalizado</option>
                  </select>
                </div>
                
                {formData.recurrence_type !== 'none' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Hora da Recorrência
                    </label>
                    <input
                      type="time"
                      value={formData.recurrence_time}
                      onChange={(e) => setFormData({ ...formData, recurrence_time: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    />
                  </div>
                )}
                
                {formData.recurrence_type === 'weekly' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Dias da Semana
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {DAYS_OF_WEEK.map(day => (
                        <label key={day.value} className="flex items-center">
                          <input
                            type="checkbox"
                            value={day.value}
                            checked={formData.recurrence_days_of_week.includes(day.value)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setFormData({
                                  ...formData,
                                  recurrence_days_of_week: [...formData.recurrence_days_of_week, day.value]
                                });
                              } else {
                                setFormData({
                                  ...formData,
                                  recurrence_days_of_week: formData.recurrence_days_of_week.filter(d => d !== day.value)
                                });
                              }
                            }}
                            className="mr-2"
                          />
                          {day.label}
                        </label>
                      ))}
                    </div>
                  </div>
                )}
                
                {formData.recurrence_type === 'monthly' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Dia do Mês
                    </label>
                    <input
                      type="number"
                      value={formData.recurrence_day_of_month}
                      onChange={(e) => setFormData({ ...formData, recurrence_day_of_month: parseInt(e.target.value) })}
                      min="1"
                      max="31"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    />
                  </div>
                )}
                
                {formData.recurrence_type === 'custom' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Intervalo em Dias
                    </label>
                    <input
                      type="number"
                      value={formData.recurrence_interval}
                      onChange={(e) => setFormData({ ...formData, recurrence_interval: parseInt(e.target.value) })}
                      min="1"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    />
                  </div>
                )}
              </div>
              
              <div className="mt-6 flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={resetForm}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  {editingSchedule ? "Atualizar" : "Criar"} Agendamento
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}