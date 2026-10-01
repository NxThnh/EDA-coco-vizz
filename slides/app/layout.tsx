import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "COCO × VizWiz — EDA cho HGAI", description: "Bộ slide EDA về domain gap và hallucination trong VLM nhỏ." };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
