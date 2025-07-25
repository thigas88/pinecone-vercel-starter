"use client";

import { useState } from "react";
import { 
  Search, 
  Filter,
  FileText,
  Tag,
  Layers,
  Hash,
  Link,
  ChevronDown,
  ChevronUp,
  X
} from "lucide-react";

interface SearchResult {
  content: string;
  metadata: {
    title?: string;
    category?: string;
    tags?: string[];
    keywords?: string[];
    url?: string;
    url_ref?: string;
    type?: string;
    chunk_index?: number;
    total_chunks?: number;
    document_id?: number;
    created_at?: string;
  };
  score: number;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  count: number;
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());
  
  // Filter states
  const [filters, setFilters] = useState({
    category: "",
    tags: "",
    k: 10,
    scoreThreshold: 0.0
  });

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        query: query,
        k: filters.k.toString(),
      });

      if (filters.category) {
        params.append("category", filters.category);
      }
      if (filters.tags) {
        params.append("tags", filters.tags);
      }
      if (filters.scoreThreshold > 0) {
        params.append("score_threshold", filters.scoreThreshold.toString());
      }

      const response = await fetch(`http://localhost:8000/search?${params}`);
      if (!response.ok) throw new Error("Search failed");
      
      const data: SearchResponse = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const toggleResultExpansion = (index: number) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedResults(newExpanded);
  };

  const highlightQuery = (text: string) => {
    if (!query) return text;
    
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part, i) => 
      part.toLowerCase() === query.toLowerCase() 
        ? <mark key={i} className="bg-yellow-200 dark:bg-yellow-800">{part}</mark>
        : part
    );
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "text-green-600 dark:text-green-400";
    if (score >= 0.6) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
          Busca Avançada
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Pesquise por conteúdo nos documentos indexados usando busca vetorial semântica
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex gap-4 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Digite sua busca..."
                className="pl-10 w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white text-lg"
                autoFocus
              />
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Buscando..." : "Buscar"}
            </button>
          </div>

          {/* Filters Toggle */}
          <button
            type="button"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
          >
            <Filter className="w-4 h-4 mr-2" />
            Filtros Avançados
            {showFilters ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
          </button>

          {/* Filters */}
          {showFilters && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Categoria
                  </label>
                  <input
                    type="text"
                    value={filters.category}
                    onChange={(e) => setFilters({ ...filters, category: e.target.value })}
                    placeholder="Ex: assinatura"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Tags (separadas por vírgula)
                  </label>
                  <input
                    type="text"
                    value={filters.tags}
                    onChange={(e) => setFilters({ ...filters, tags: e.target.value })}
                    placeholder="Ex: tag1, tag2"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Número de Resultados
                  </label>
                  <input
                    type="number"
                    value={filters.k}
                    onChange={(e) => setFilters({ ...filters, k: parseInt(e.target.value) || 10 })}
                    min="1"
                    max="50"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Score Mínimo (0-1)
                  </label>
                  <input
                    type="number"
                    value={filters.scoreThreshold}
                    onChange={(e) => setFilters({ ...filters, scoreThreshold: parseFloat(e.target.value) || 0 })}
                    min="0"
                    max="1"
                    step="0.1"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </form>

      {/* Error Message */}
      {error && (
        <div className="mb-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
              {results.length} resultado{results.length !== 1 ? 's' : ''} encontrado{results.length !== 1 ? 's' : ''}
            </h2>
            <button
              onClick={() => setResults([])}
              className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
            >
              <X className="w-4 h-4 inline mr-1" />
              Limpar resultados
            </button>
          </div>

          {results.map((result, index) => {
            const isExpanded = expandedResults.has(index);
            const contentPreview = result.content.substring(0, 200) + (result.content.length > 200 ? "..." : "");
            
            return (
              <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg transition-shadow">
                <div className="p-6">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
                        {result.metadata.title || `Documento #${result.metadata.document_id}`}
                      </h3>
                      <div className="flex flex-wrap gap-2 text-sm">
                        {result.metadata.category && (
                          <span className="flex items-center text-gray-600 dark:text-gray-400">
                            <Layers className="w-4 h-4 mr-1" />
                            {result.metadata.category}
                          </span>
                        )}
                        {result.metadata.chunk_index !== undefined && (
                          <span className="flex items-center text-gray-600 dark:text-gray-400">
                            <Hash className="w-4 h-4 mr-1" />
                            Chunk {result.metadata.chunk_index + 1} de {result.metadata.total_chunks}
                          </span>
                        )}
                        {result.metadata.type && (
                          <span className="flex items-center text-gray-600 dark:text-gray-400">
                            <FileText className="w-4 h-4 mr-1" />
                            {result.metadata.type}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="ml-4">
                      <span className={`text-2xl font-bold ${getScoreColor(result.score)}`}>
                        {formatScore(result.score)}%
                      </span>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Relevância</p>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="mb-4">
                    <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                      {isExpanded ? (
                        <span>{highlightQuery(result.content)}</span>
                      ) : (
                        <span>{highlightQuery(contentPreview)}</span>
                      )}
                    </p>
                    {result.content.length > 200 && (
                      <button
                        onClick={() => toggleResultExpansion(index)}
                        className="mt-2 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        {isExpanded ? "Mostrar menos" : "Mostrar mais"}
                      </button>
                    )}
                  </div>

                  {/* Metadata Footer */}
                  <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex flex-wrap gap-4 text-sm">
                      {result.metadata.url && (
                        <a
                          href={result.metadata.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          <Link className="w-4 h-4 mr-1" />
                          Ver documento original
                        </a>
                      )}
                      {result.metadata.tags && result.metadata.tags.length > 0 && (
                        <div className="flex items-center gap-2">
                          <Tag className="w-4 h-4 text-gray-400" />
                          {result.metadata.tags.map((tag, i) => (
                            <span key={i} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!loading && query && results.length === 0 && !error && (
        <div className="text-center py-12">
          <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Nenhum resultado encontrado para &quot;{query}&quot;
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            Tente usar termos diferentes ou ajustar os filtros
          </p>
        </div>
      )}

      {/* Initial State */}
      {!loading && !query && results.length === 0 && (
        <div className="text-center py-12">
          <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Digite algo para começar a buscar
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            A busca utiliza similaridade semântica para encontrar conteúdo relevante
          </p>
        </div>
      )}
    </div>
  );
}