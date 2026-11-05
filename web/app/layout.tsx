import type {Metadata} from 'next';
import './globals.css';
export const metadata:Metadata={title:'CourtVision · NBA Analytics',description:'Appearance-conditional NBA forecasts, transparent provenance and forward tracking.'};
export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
