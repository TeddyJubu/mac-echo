// The page shown at the top of the MacEcho window: the orb, with frequency bars under it.
// The Python app (macecho/menubar.py) drives it by calling
//   window.macecho(state, mic, speaker, bands)
// where state is its state name, mic and speaker are loudness from 0 to 1, and bands is the
// loudness of each frequency band, low to high. Levels arrive about 20 times a second while it
// listens. Python decides how many bands there are; the page draws one bar per band.
import { useSyncExternalStore } from "react"
import { createRoot } from "react-dom/client"

import { BarVisualizer, type AgentState } from "./bar-visualizer"
import "./index.css"
import { Orb } from "./orb"

declare global {
  interface Window {
    macecho: (state: string, mic: number, speaker: number, bands: number[]) => void
  }
}

const COLORS: [string, string] = ["#CADCFC", "#A0B9D1"]
const ERROR_COLORS: [string, string] = ["#FCA5A5", "#EF4444"]
const IDLE_MOTION = 0.3 // the output level the orb uses for its own resting movement
// What the bars do in each app state: sweep while the model loads, blink the centre bar while
// listening, light up fully while replying. Off and error have no entry: a still row of dots.
const BAR_STATES: Record<string, AgentState> = {
  loading: "connecting",
  listening: "listening",
  replying: "speaking",
}

let state = "loading"
let bands: number[] = []
const listeners = new Set<() => void>()
// Plain refs: the orb reads them every frame, so a new level never re-renders it.
const input = { current: 0 }
const output = { current: IDLE_MOTION }

window.macecho = (nextState, mic, speaker, nextBands) => {
  input.current = Math.max(mic, speaker) // the rings swell with whoever is speaking
  output.current = IDLE_MOTION + (1 - IDLE_MOTION) * speaker // the surface churns with the reply
  state = nextState
  bands = nextBands
  listeners.forEach((notify) => notify())
}

function subscribe(notify: () => void) {
  listeners.add(notify)
  return () => listeners.delete(notify)
}

// Two components so that 20 band updates a second re-render only the bars: the orb subscribes
// to the state name alone, which rarely changes.
function OrbView() {
  const current = useSyncExternalStore(subscribe, () => state)
  const live = current === "listening" || current === "replying"
  return (
    <Orb
      className="size-[150px]"
      colors={current === "error" ? ERROR_COLORS : COLORS}
      // Live: follow the real levels. Otherwise the orb animates itself: "thinking" while the
      // model loads, at rest when off or failed.
      volumeMode={live ? "manual" : "auto"}
      agentState={current === "loading" ? "thinking" : null}
      inputVolumeRef={input}
      outputVolumeRef={output}
    />
  )
}

function Bars() {
  const current = useSyncExternalStore(subscribe, () => state)
  const levels = useSyncExternalStore(subscribe, () => bands)
  return (
    <BarVisualizer
      className="h-10 gap-1 rounded-none bg-transparent p-0"
      state={BAR_STATES[current]}
      barCount={levels.length}
      bands={levels}
      centerAlign
    />
  )
}

// Both components shade for dark mode when <html> has the "dark" class; follow the system.
const dark = matchMedia("(prefers-color-scheme: dark)")
const applyTheme = () => document.documentElement.classList.toggle("dark", dark.matches)
applyTheme()
dark.addEventListener("change", applyTheme)

createRoot(document.getElementById("root")!).render(
  <div className="flex h-full flex-col items-center gap-2">
    <OrbView />
    <Bars />
  </div>
)
