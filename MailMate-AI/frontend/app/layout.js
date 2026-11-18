import { Geist, Geist_Mono } from "next/font/google";
import Footer from "../components/Footer";
import Navbar from "../components/Navbar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "MailMate — Context-Aware Email Text Prediction",
  description:
    "Real-time next-word and prefix completion for professional email, powered by interpolated N-gram language models, WordNet tone scoring, and TF-IDF context classification — running locally on CPU.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body>
        <Navbar />
        <main className="site-main">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
