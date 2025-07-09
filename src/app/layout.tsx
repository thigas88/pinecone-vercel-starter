import { Analytics } from "@vercel/analytics/react"
import { Inter } from "next/font/google";
import "../global.css";
import { ThemeProvider } from "@/components/theme-provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Suporte AI - STI",
  description: "Assistente de Suporte AI para os serviços e sistemas da STI/UFVJM.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <Analytics/>
      <body className={`${inter.className} bg-muted/30`}>
        <ThemeProvider attribute="class" defaultTheme="system">
        {children}
        </ThemeProvider>        
      </body>
    </html>
  );
}
