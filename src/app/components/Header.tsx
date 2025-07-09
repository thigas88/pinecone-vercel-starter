
export default function Header({ className }: { className?: string }) {
  return (
    <header
      className={`flex items-center justify-center text-gray-600 text-2xl ${className}`}
    >
      Suporte Inteligente da STI
    </header>
  );
}
