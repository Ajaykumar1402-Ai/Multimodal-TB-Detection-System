/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0f1c', // Dark modern blue
        surface: '#121b2d', // Slightly lighter
        primary: '#38bdf8', // Neon cyan
        secondary: '#818cf8', // Soft indigo
        textmain: '#f1f5f9',
        textmuted: '#94a3b8'
      }
    },
  },
  plugins: [],
}
