"use client";

import { useEffect, useState } from "react";
import { 
  FileText, 
  Plus, 
  RefreshCw, 
  Trash2, 
  Eye,
  Download,
  Upload,
  Link,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Search,
  Filter,
  Globe
} from "lucide-react";

interface Document {
  id: number;
  url: string;
  title: string | null;
  tags: string | null;
  category: string | null;
  splitting_method: string | null;
  chunk_size: number | null;
  overlap: number | null;
  file_name: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  scheduled_at: string | null;
  last_indexed_at: string | null;
}

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

interface Category {
  id: number;
  name: string;
  description: string | null;
  color: string | null;
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [history, setHistory] = useState<IngestHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterCategory, setFilterCategory] = useState<string>("all");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch documents
      const docsRes = await fetch("http://localhost:8000/admin/documents");
      if (!docsRes.ok) throw new Error("Failed to fetch documents");
      const docsData = await docsRes.json();
      setDocuments(docsData);
      
      // Fetch categories
      const catRes = await fetch("http://localhost:8000/admin/categories");
      if (!catRes.ok) throw new Error("Failed to fetch categories");
      const catData = await catRes.json();
      setCategories(catData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const fetchDocuments = fetchData;

  const fetchHistory = async (docId: number) => {
    try {
      const res = await fetch(`http://localhost:8000/admin/documents/${docId}/history`);
      if (!res.ok) throw new Error("Failed to fetch history");
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    }
  };

  const handleReindex = async (doc: Document) => {
    if (!confirm(`Tem certeza que deseja reindexar o documento "${doc.title || doc.url}"?`)) {
      return;
    }

    try {
      // Use the new reindex endpoint
      const response = await fetch(`http://localhost:8000/admin/documents/${doc.id}/reindex`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to reindex document");
      }

      const result = await response.json();
      alert(`Reindexação iniciada com sucesso! ${result.message}`);
      fetchDocuments();
    } catch (err) {
      alert(`Erro ao reindexar: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  const handlePreviewContent = async (doc: Document) => {
    try {
      const response = await fetch(`http://localhost:8000/admin/documents/${doc.id}/fetch-content`);
      if (!response.ok) {
        throw new Error("Failed to fetch content");
      }

      const result = await response.json();
      if (result.status === "error") {
        alert(`Erro ao buscar conteúdo: ${result.error}`);
      } else {
        alert(`Conteúdo encontrado!\n\nTítulo: ${result.title || 'N/A'}\nTamanho: ${result.content_length} caracteres\n\nPrévia:\n${result.content_preview}`);
      }
    } catch (err) {
      alert(`Erro ao buscar conteúdo: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  const handleViewHistory = (doc: Document) => {
    setSelectedDoc(doc);
    fetchHistory(doc.id);
    setShowHistoryModal(true);
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleString("pt-BR");
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "indexed":
      case "completed":
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case "processing":
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case "error":
        return <XCircle className="w-4 h-4 text-red-500" />;
      case "agendado":
        return <Clock className="w-4 h-4 text-blue-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
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

  // Filter documents
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         doc.url.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         doc.category?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === "all" || doc.status === filterStatus;
    const matchesCategory = filterCategory === "all" || doc.category === filterCategory;
    
    return matchesSearch && matchesStatus && matchesCategory;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Carregando documentos...</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white">
          Gerenciamento de Documentos
        </h1>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus className="w-5 h-5 mr-2" />
          Adicionar Documento
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Buscar
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Buscar por título, URL ou categoria..."
                className="pl-10 w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Status
            </label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="all">Todos</option>
              <option value="indexed">Indexado</option>
              <option value="processing">Processando</option>
              <option value="error">Erro</option>
              <option value="agendado">Agendado</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Categoria
            </label>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="all">Todas</option>
              {categories.map(cat => (
                <option key={cat.id} value={cat.name}>{cat.name}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Documents Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Documento
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Categoria
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Última Indexação
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Método de Split
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Ações
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {filteredDocuments.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-4 text-center text-gray-500 dark:text-gray-400">
                    Nenhum documento encontrado
                  </td>
                </tr>
              ) : (
                filteredDocuments.map((doc) => (
                  <tr key={doc.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="w-5 h-5 text-gray-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {doc.title || `Documento #${doc.id}`}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {doc.url.length > 50 ? doc.url.substring(0, 50) + "..." : doc.url}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-gray-900 dark:text-white">
                        {doc.category || "-"}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {getStatusIcon(doc.status)}
                        <span className="ml-2">{getStatusBadge(doc.status)}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatDate(doc.last_indexed_at)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {doc.splitting_method || "character"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end space-x-2">
                        <button
                          onClick={() => handleViewHistory(doc)}
                          className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                          title="Ver histórico"
                        >
                          <Eye className="w-5 h-5" />
                        </button>
                        {(doc.url.startsWith('http://') || doc.url.startsWith('https://')) && (
                          <>
                            <button
                              onClick={() => handlePreviewContent(doc)}
                              className="text-purple-600 hover:text-purple-900 dark:text-purple-400 dark:hover:text-purple-300"
                              title="Visualizar conteúdo da web"
                            >
                              <Globe className="w-5 h-5" />
                            </button>
                            <button
                              onClick={() => handleReindex(doc)}
                              className="text-green-600 hover:text-green-900 dark:text-green-400 dark:hover:text-green-300"
                              title="Reindexar documento"
                            >
                              <RefreshCw className="w-5 h-5" />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* History Modal */}
      {showHistoryModal && selectedDoc && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden">
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
                Histórico de Processamento
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {selectedDoc.title || selectedDoc.url}
              </p>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {history.length === 0 ? (
                <p className="text-gray-500 dark:text-gray-400">Nenhum histórico encontrado</p>
              ) : (
                <div className="space-y-4">
                  {history.map((item) => (
                    <div key={item.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center">
                          {getStatusIcon(item.status)}
                          <div className="ml-3">
                            <p className="font-medium text-gray-800 dark:text-white">
                              {item.message || `Processamento #${item.id}`}
                            </p>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              Iniciado: {formatDate(item.started_at)}
                            </p>
                            {item.finished_at && (
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                Finalizado: {formatDate(item.finished_at)}
                              </p>
                            )}
                            {item.chunks_indexed && (
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                Chunks indexados: {item.chunks_indexed}
                              </p>
                            )}
                          </div>
                        </div>
                        {getStatusBadge(item.status)}
                      </div>
                      {item.error && (
                        <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 rounded">
                          <p className="text-sm text-red-800 dark:text-red-200">
                            Erro: {item.error}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-6 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setShowHistoryModal(false)}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Fechar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Document Modal */}
      {showAddModal && (
        <AddDocumentModal
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false);
            fetchDocuments();
          }}
          categories={categories}
        />
      )}
    </div>
  );
}

// Add Document Modal Component
function AddDocumentModal({
  onClose,
  onSuccess,
  categories
}: {
  onClose: () => void;
  onSuccess: () => void;
  categories: Category[];
}) {
  const [formData, setFormData] = useState({
    url: "",
    title: "",
    category: "",
    tags: "",
    splitting_method: "character",
    chunk_size: 1000,
    chunk_overlap: 200,
    content: "",
    file_type: "web"
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingStarted, setProcessingStarted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/admin/ingest/document", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...formData,
          tags: formData.tags ? formData.tags.split(',').map(t => t.trim()) : [],
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to add document");
      }

      const result = await response.json();
      setProcessingStarted(true);
      
      // Show success message with processing info
      alert(`Documento adicionado com sucesso!\n\nID: ${result.document_id}\nStatus: ${result.status}\n\nO processamento está sendo executado em background.`);
      
      // Close modal after a short delay
      setTimeout(() => {
        onSuccess();
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
            Adicionar Novo Documento
          </h2>
        </div>
        
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[70vh]">
          {error && (
            <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                URL do Documento
              </label>
              <input
                type="text"
                value={formData.url}
                onChange={(e) => setFormData({ ...formData, url: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                placeholder="https://exemplo.com/documento"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Título
              </label>
              <input
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Categoria
              </label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="">Selecione uma categoria</option>
                {categories.map(cat => (
                  <option key={cat.id} value={cat.name}>{cat.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Tags (separadas por vírgula)
              </label>
              <input
                type="text"
                value={formData.tags}
                onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                placeholder="tag1, tag2, tag3"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Método de Divisão
              </label>
              <select
                value={formData.splitting_method}
                onChange={(e) => setFormData({ ...formData, splitting_method: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="character">Caracteres</option>
                <option value="sentence">Sentenças</option>
                <option value="semantic">Semântico</option>
                <option value="textsemantic">Semântico Aprimorado</option>
                <option value="markdown">Markdown</option>
              </select>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Tamanho do Chunk
                </label>
                <input
                  type="number"
                  value={formData.chunk_size}
                  onChange={(e) => setFormData({ ...formData, chunk_size: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  min="100"
                  max="10000"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Sobreposição
                </label>
                <input
                  type="number"
                  value={formData.chunk_overlap}
                  onChange={(e) => setFormData({ ...formData, chunk_overlap: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  min="0"
                  max="1000"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Conteúdo do Documento
              </label>
              <textarea
                value={formData.content}
                onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                rows={6}
                placeholder="Cole o conteúdo do documento aqui..."
              />
            </div>
          </div>
          
          {processingStarted && (
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <p className="text-blue-800 dark:text-blue-200 flex items-center">
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Processamento iniciado em background...
              </p>
            </div>
          )}
          
          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              disabled={loading}
            >
              Cancelar
            </button>
            <button
              type="submit"
              disabled={loading || processingStarted}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {loading ? "Adicionando..." : "Adicionar Documento"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}