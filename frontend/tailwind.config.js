/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        hdfc: {
          blue: '#004C8F',       // HDFC Special Blue (Primary)
          red: '#ED232A',        // HDFC Funky Red (Secondary / Accent)
          dark: '#002D56',       // HDFC Deep Navy (Header / High-contrast)
          darker: '#001E3D',     // Deep Corporate Slate
          light: '#F0F6FC',      // Soft Banking Tint
          hover: '#003A70',      // Darker Blue for hover states
          redHover: '#D91B22',   // Darker Red for hover states
          border: '#CBDCEE',     // Subtle Blue-Gray Border
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
