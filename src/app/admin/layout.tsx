import { Inter } from "next/font/google";
import Link from "next/link";
import { 
  FileText, 
  Activity, 
  Calendar, 
  BarChart3, 
  Settings,
  Home,
  Search
} from "lucide-react";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Painel Administrativo - ChatRAG",
  description: "Gerenciamento de documentos e processamentos do ChatRAG",
};

const navItems = [
  { href: "/admin", label: "Dashboard", icon: Home },
  { href: "/admin/documents", label: "Documentos", icon: FileText },
  { href: "/admin/processing", label: "Processamentos", icon: Activity },
  { href: "/admin/schedules", label: "Agendamentos", icon: Calendar },
  { href: "/admin/statistics", label: "Estatísticas", icon: BarChart3 },
  { href: "/admin/search", label: "Buscar", icon: Search },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <aside className="w-64 bg-white dark:bg-gray-800 shadow-md">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-white">
            ChatRAG Admin
          </h1>
        </div>
        
        <nav className="mt-6">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center px-6 py-3 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <Icon className="w-5 h-5 mr-3" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        
        <div className="absolute bottom-0 w-64 p-6">
          <Link
            href="/"
            className="flex items-center text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
          >
            <Settings className="w-5 h-5 mr-2" />
            <span>Voltar ao Chat</span>
          </Link>
        </div>
      </aside>
      
      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
}