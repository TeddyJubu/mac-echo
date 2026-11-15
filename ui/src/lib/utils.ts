// The class-name helper every shadcn/ui component imports: joins class names and lets a later
// Tailwind class override an earlier conflicting one (so className="h-10" beats a built-in h-32).
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
