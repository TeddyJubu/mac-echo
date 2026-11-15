// The whole MacEcho window: the orb, frequency bars, what the assistant is doing, the
// Start/Stop control and the voice switch. The Python app (macecho/menubar.py) drives it with
//   window.macecho(state, mic, speaker, bands)   ~20 times a second while a session runs
//   window.macechoInfo(info)                     ~3 times a second (see Info below)
// and hears clicks back as webkit.messageHandlers.macecho.postMessage("toggle" | "model:<name>").
import { useEffect, useSyncExternalStore } from "react"
import { createRoot } from "react-dom/client"

import { BarVisualizer, type AgentState } from "./bar-visualizer"
import "./index.css"
import { cn } from "./lib/utils"
import { Orb } from "./orb"

type Info = {
  error: string | null // "Model failed to load: …" or "Agent stopped: …"
  running: boolean // a listening session is on
  pending: boolean // Start was pressed while the model loads; it starts when ready
  busy: boolean // a model is loading; switching is ignored meanwhile
  models: string[]
  model: string | null // null: a model set in .env that isn't one of `models`
  viewer?: boolean // menubar --view: only showing Claude Code's voice mode, so no controls
  connected?: boolean // viewer: a macecho daemon is running
}

declare global {
  interface Window {
    macecho: (state: string, mic: number, speaker: number, bands: number[]) => void
    macechoInfo: (info: Info) => void
    webkit?: { messageHandlers: { macecho: { postMessage: (message: string) => void } } }
  }
}

// One colour world per state. The orb ramps black → orb[0] → orb[1] → white; the accent tints
// the glow behind it, the bars and focus rings, and glides between states (see index.css).
const THEMES: Record<string, { orb: [string, string]; accent: string }> = {
  loading: { orb: ["#8A93A8", "#B4BCCE"], accent: "oklch(0.74 0.03 262)" },
  off: { orb: ["#5E6578", "#8D95A8"], accent: "oklch(0.6 0.015 262)" },
  listening: { orb: ["#CADCFC", "#A0B9D1"], accent: "oklch(0.86 0.07 252)" },
  replying: { orb: ["#F3DAFF", "#B98AF0"], accent: "oklch(0.8 0.13 305)" },
  error: { orb: ["#FCA5A5", "#EF4444"], accent: "oklch(0.72 0.17 25)" },
}
const IDLE_MOTION = 0.3 // the output level the orb uses for its own resting movement
// Sweep while the model loads, blink the centre bar while listening, light up while replying.
const BAR_STATES: Record<string, AgentState> = {
  loading: "connecting",
  listening: "listening",
  replying: "speaking",
}

let state = "loading"
let bands: number[] = []
let info: Info = { error: null, running: false, pending: false, busy: true, models: [], model: null }
let infoJson = ""
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

window.macechoInfo = (next) => {
  const json = JSON.stringify(next)
  if (json === infoJson) return // sent on a timer; most calls change nothing
  infoJson = json
  info = next
  listeners.forEach((notify) => notify())
}

function subscribe(notify: () => void) {
  listeners.add(notify)
  return () => listeners.delete(notify)
}

const useAppState = () => useSyncExternalStore(subscribe, () => state)
const useInfo = () => useSyncExternalStore(subscribe, () => info)
const send = (message: string) => window.webkit?.messageHandlers.macecho.postMessage(message)

// What the window says in each state: a headline and the line under it.
function describe(current: string, { error, model, viewer, connected }: Info) {
  if (viewer) {
    switch (current) {
      case "listening":
        return { title: "Listening", detail: "Claude Code is waiting for you" }
      case "replying":
        return { title: "Speaking", detail: "Claude Code is answering" }
      default:
        return connected
          ? { title: "Voice mode", detail: "Claude Code speaks when it has something to say" }
          : { title: "Voice mode is off", detail: "Say “voice mode” in Claude Code to start" }
    }
  }
  if (current === "error" && error) {
    const cut = error.indexOf(": ") // "Model failed to load: why" → headline, then the why
    return cut < 0
      ? { title: error, detail: "" }
      : { title: error.slice(0, cut), detail: error.slice(cut + 2) }
  }
  switch (current) {
    case "loading":
      return { title: "Loading model", detail: model ?? "Custom voice" }
    case "listening":
      return { title: "Listening", detail: "Go ahead and speak" }
    case "replying":
      return { title: "Replying", detail: "Press Return to stop" }
    default:
      return { title: "Not listening", detail: "Press Return to start" }
  }
}

// Two components for the visuals so that 20 band updates a second re-render only the bars: the
// orb subscribes to the state name alone, which rarely changes.
function OrbStage() {
  const current = useAppState()
  const live = current === "listening" || current === "replying"
  return (
    <div className="relative grid h-[212px] place-items-center" data-rest={!live && current !== "error"}>
      <div aria-hidden className="glow absolute inset-[-40px]" data-live={live} />
      <Orb
        className="orb relative size-[188px]"
        colors={(THEMES[current] ?? THEMES.off).orb}
        // Live: follow the real levels. Otherwise the orb animates itself: "thinking" while the
        // model loads, at rest when off or failed.
        volumeMode={live ? "manual" : "auto"}
        agentState={current === "loading" ? "thinking" : null}
        inputVolumeRef={input}
        outputVolumeRef={output}
      />
    </div>
  )
}

function Bars() {
  const current = useAppState()
  const levels = useSyncExternalStore(subscribe, () => bands)
  return (
    <BarVisualizer
      aria-hidden
      className="h-8 w-[164px] gap-[5px] rounded-none bg-transparent p-0"
      state={BAR_STATES[current]}
      barCount={levels.length}
      bands={levels}
      minHeight={25}
      centerAlign
    />
  )
}

function Status() {
  const { title, detail } = describe(useAppState(), useInfo())
  return (
    // key: a new headline mounts fresh, so it plays the entrance instead of snapping.
    <div role="status" aria-live="polite" className="flex min-h-[62px] flex-col items-center px-7 text-center">
      <h1 key={title} className="rise text-[21px] leading-7 font-[580] tracking-[-0.022em] text-white/95">
        {title}
      </h1>
      <p
        key={`${title}/${detail}`}
        title={detail}
        className="rise line-clamp-2 text-[13px] leading-[18px] text-[color-mix(in_oklch,var(--accent)_45%,white_40%)] [animation-delay:60ms] select-text"
      >
        {detail}
      </p>
    </div>
  )
}

function Primary() {
  const current = useAppState()
  const { running, pending, busy, viewer } = useInfo()
  if (viewer) return <div className="h-11" /> // Claude Code decides when to listen, not a button
  const [label, icon, strong] = running
    ? ["Stop", <StopIcon />, false]
    : pending && busy
      ? ["Starting…", <SpinnerIcon />, false]
      : current === "error"
        ? ["Try again", <RetryIcon />, true]
        : ["Start listening", <MicIcon />, true]
  return (
    <button
      type="button"
      onClick={() => send("toggle")}
      className={cn(
        "inline-flex h-11 min-w-[176px] items-center justify-center gap-2 rounded-full px-6",
        "text-[14px] font-[560] tracking-[-0.01em] transition-[background-color,color,scale] duration-200",
        "focus-visible:outline-2 focus-visible:outline-offset-[3px] focus-visible:outline-[var(--accent)]",
        "active:scale-[0.97]",
        strong
          ? "bg-white/[0.93] text-[#0A0B0E] shadow-[0_6px_20px_-6px_color-mix(in_oklch,var(--accent)_55%,transparent)] hover:bg-white"
          : "bg-white/[0.08] text-white/90 hover:bg-white/[0.12]"
      )}
    >
      {icon}
      {label}
    </button>
  )
}

function ModelSwitch() {
  const { models, model, busy } = useInfo()
  const index = model === null ? -1 : models.indexOf(model)
  if (!models.length) return <div className="h-8" />
  return (
    <div
      role="radiogroup"
      aria-label="Voice"
      title={busy ? "Switch voices once loading finishes" : "Voice"}
      className={cn(
        "relative grid h-8 w-[256px] rounded-full bg-white/[0.05] p-[3px] transition-opacity duration-300",
        busy && "opacity-45"
      )}
      style={{ gridTemplateColumns: `repeat(${models.length}, 1fr)` }}
    >
      <span
        aria-hidden
        className="segment-thumb absolute inset-y-[3px] left-[3px] rounded-full bg-white/[0.11]"
        style={{
          width: `calc((100% - 6px) / ${models.length})`,
          translate: `${Math.max(index, 0) * 100}% 0`,
          opacity: index < 0 ? 0 : 1,
        }}
      />
      {models.map((name, i) => (
        <label
          key={name}
          className={cn(
            "relative z-10 grid place-items-center rounded-full text-[12px] font-[520] tracking-[-0.005em]",
            "transition-colors duration-200 has-focus-visible:outline-2 has-focus-visible:outline-[var(--accent)]",
            i === index ? "text-white/95" : "text-white/45",
            !busy && i !== index && "hover:text-white/75"
          )}
        >
          <input
            type="radio"
            name="model"
            className="sr-only"
            checked={i === index}
            disabled={busy}
            onChange={() => send(`model:${name}`)}
          />
          {name}
        </label>
      ))}
    </div>
  )
}

// Icons: 16 pt, drawn on a 24 grid with one stroke weight.
const Icon = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <svg
    aria-hidden
    viewBox="0 0 24 24"
    className={cn("size-4 shrink-0", className)}
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {children}
  </svg>
)
const MicIcon = () => (
  <Icon>
    <rect x="9" y="2.5" width="6" height="12" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5v4" />
  </Icon>
)
const StopIcon = () => (
  <Icon>
    <rect x="6.5" y="6.5" width="11" height="11" rx="2.5" fill="currentColor" stroke="none" />
  </Icon>
)
const RetryIcon = () => (
  <Icon>
    <path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1L3.5 8.5M3.5 3.5v5h5" />
  </Icon>
)
const SpinnerIcon = () => (
  <Icon className="animate-spin [animation-duration:900ms]">
    <path d="M20.5 12a8.5 8.5 0 1 1-6-8.13" />
  </Icon>
)

function App() {
  const current = useAppState()
  useEffect(() => {
    // Everything tinted reads --accent, a registered colour, so the whole window glides.
    document.documentElement.style.setProperty("--accent", (THEMES[current] ?? THEMES.off).accent)
  }, [current])
  return (
    <main className="flex h-full flex-col items-center pt-1 pb-6">
      <OrbStage />
      <Bars />
      <div className="mt-5">
        <Status />
      </div>
      <div className="mt-4">
        <Primary />
      </div>
      <div className="mt-auto">
        <ModelSwitch />
      </div>
    </main>
  )
}

// Return toggles listening from anywhere; on the button itself the button's own click does it.
addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.repeat || event.target instanceof HTMLButtonElement) return
  event.preventDefault()
  send("toggle")
})
addEventListener("contextmenu", (event) => event.preventDefault()) // no "Reload" in an app window

createRoot(document.getElementById("root")!).render(<App />)
