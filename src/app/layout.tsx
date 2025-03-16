import { Analytics } from "@vercel/analytics/react"

export const metadata = {
  title: "Suporte AI - STI",
  description: "Assistente de Suporte AI para os serviços e sistemas da STI/UFVJM.",
};

import "../global.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <Analytics/>
      <body>{children}</body>
    </html>
  );
}
