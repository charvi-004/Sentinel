import type { Metadata } from 'next'
import './globals.css'
export const metadata: Metadata={title:'Sentinel Risk Engine',description:'Fraud risk analysis for operations teams'}
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en" className="bg-[#F7F9FC]"><body>{children}</body></html>}
