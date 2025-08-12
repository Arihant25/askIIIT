import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "askIIIT",
  description: "Ask questions about IIIT documents and get instant answers",
  keywords: [
    "IIIT",
    "documents",
    "Q&A",
    "query",
    "information",
    "PDF",
    "search",
  ],
  openGraph: {
    title: "askIIIT",
    description: "Ask questions about IIIT documents and get instant answers",
    siteName: "askIIIT",
    locale: "en_US",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Navbar />
        {children}
      </body>
    </html>
  );
}
