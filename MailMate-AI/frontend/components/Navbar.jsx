"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/how-it-works", label: "How It Works" },
  { href: "/about", label: "About" },
];

function MailCursorMark() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="2.25" y="4.5" width="13.5" height="10" rx="1.75" />
      <path d="M2.75 5.5 9 10l6.25-4.5" />
      <path d="M14.5 15.5 16.25 17 18 13.5" />
    </svg>
  );
}

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <div className="container header-inner">
        <Link href="/" className="brand" aria-label="MailMate home">
          <span className="brand-mark">
            <MailCursorMark />
          </span>
          <span>MailMate</span>
          <span className="brand-version">v1.0</span>
        </Link>

        <nav className="nav-links" aria-label="Primary">
          {LINKS.map((link) => {
            const isActive =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`nav-link${isActive ? " is-active" : ""}`}
                aria-current={isActive ? "page" : undefined}
              >
                {link.label}
              </Link>
            );
          })}
          <span className="nav-cta">
            <Link href="/studio" className="btn btn-primary btn-sm">
              Open Studio <span aria-hidden="true">→</span>
            </Link>
          </span>
        </nav>
      </div>
    </header>
  );
}
