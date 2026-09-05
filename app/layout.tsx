import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const jetbrains = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' })

export const metadata: Metadata = { title: 'Sentinel Risk Engine', description: 'Fraud risk analysis for operations teams' }
export default function RootLayout({ children }: { children: React.ReactNode }) { return <html lang="en" className="bg-canvas"><body className={`${inter.variable} ${jetbrains.variable}`}>{children}</body></html> }
