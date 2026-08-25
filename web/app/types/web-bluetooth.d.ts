// Ambient Web Bluetooth types -- Nuxt's generated tsconfig sets
// compilerOptions.types: [] (so @types/* packages are never auto-included),
// so this triple-slash reference is the only thing pulling
// @types/web-bluetooth's global Navigator.bluetooth augmentation into the
// project. Used by app/pages/audio-debug.vue's real (non-simulated)
// Bluetooth device scan.
/// <reference types="web-bluetooth" />
