import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(
    'https://dwaipayanhaldar.github.io/spectrefaces',
  ),
  title: 'Eigenfaces vs SpectreFaces — Face It: It’s Just Linear Algebra',
  description:
    'Explore a face reconstructed as an average plus 24 adjustable principal components in a looping linear carousel.',
  openGraph: {
    title: 'Eigenfaces vs SpectreFaces',
    description: 'Face It: It’s Just Linear Algebra',
    type: 'website',
    images: [
      {
        url: '/og.png',
        width: 1731,
        height: 909,
        alt: 'Eigenfaces vs SpectreFaces — Face It: It’s Just Linear Algebra',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Eigenfaces vs SpectreFaces',
    description: 'Face It: It’s Just Linear Algebra',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
