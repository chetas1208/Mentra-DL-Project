import type { Config } from 'tailwindcss'

export default <Partial<Config>>{
  theme: {
    extend: {
      colors: {
        panel: {
          bg: '#0B1116',
          surface: '#131B22',
          line: '#223039',
        },
        wearer: {
          DEFAULT: '#5EEAD4',
          dim: '#1F4E48',
        },
        environment: {
          DEFAULT: '#FB923C',
          dim: '#5A3818',
        },
        ink: {
          DEFAULT: '#E7EEF2',
          dim: '#7C8A93',
          faint: '#4B5860',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
}
