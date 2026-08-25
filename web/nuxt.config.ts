// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['@nuxtjs/tailwindcss'],
  runtimeConfig: {
    public: {
      // ws:// for local dev, wss:// once behind Cloudflare Tunnel/TLS.
      // Override via NUXT_PUBLIC_BACKEND_WS_URL at build/deploy time.
      backendWsUrl: 'ws://127.0.0.1:8765',
    },
  },
  app: {
    head: {
      title: 'MentraWearNet',
      meta: [
        // dark color-scheme so native form controls (the input-device
        // select, on mobile especially) render dark-themed instead of a
        // jarring light-mode dropdown against this page's dark panel.
        { name: 'color-scheme', content: 'dark' },
        { name: 'theme-color', content: '#0B1116' },
        // Link-preview card for WhatsApp/Discord/iMessage/etc -- absolute
        // URL required, relative paths don't render in most unfurlers.
        { property: 'og:title', content: 'MentraWearNet' },
        { property: 'og:description', content: 'Wearer-vs-environment speech detection, live.' },
        { property: 'og:type', content: 'website' },
        { property: 'og:image', content: 'https://mentra-dl-project.vercel.app/og-image.png' },
        { property: 'og:image:width', content: '1200' },
        { property: 'og:image:height', content: '630' },
        { name: 'twitter:card', content: 'summary_large_image' },
        { name: 'twitter:title', content: 'MentraWearNet' },
        { name: 'twitter:description', content: 'Wearer-vs-environment speech detection, live.' },
        { name: 'twitter:image', content: 'https://mentra-dl-project.vercel.app/og-image.png' },
      ],
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: 'anonymous' },
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap',
        },
      ],
    },
  },
})
