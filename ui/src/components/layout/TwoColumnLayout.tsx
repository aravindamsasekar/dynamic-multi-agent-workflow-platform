import type { ReactNode } from 'react'

interface TwoColumnLayoutProps {
  left: ReactNode
  right: ReactNode
}

export default function TwoColumnLayout({ left, right }: TwoColumnLayoutProps) {
  return (
    <div className="two-column-layout">
      <div className="left-column">{left}</div>
      <div className="right-column">{right}</div>
    </div>
  )
}
