"use client";

import {
  Bell,
  Check,
  ChevronDown,
  ChevronRight,
  LogOut,
  Plus,
  Search,
  Settings,
  User,
} from "lucide-react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect, useState } from "react";

import { api, type Athlete } from "@/lib/api";
import { sections } from "@/lib/sections";
import { Logo } from "@/components/ui/Logo";
import { useAppLocale, type AppLocale } from "@/lib/appLocale";

const shellCopy = {
  uk: {
    tagline: "Тріатлонна аналітика",
    workspace: "Kachamba Lab",
    swimming: "Плавання",
    running: "Біг",
    cycling: "Велоспорт",
    dryland: "Сухе тренування",
    rehabilitation: "Реабілітація",
    history: "Історія",
    assistant: "AI асистент",
    tools: "Інструменти",
    athletes: "Атлети",
    search: "Пошук сесій і вправ…",
    loading: "Завантаження…",
    switchAthlete: "Змінити атлета",
    addAthlete: "Додати атлета",
    profile: "Профіль",
    settings: "Налаштування",
    signOut: "Вийти",
    notifications: "Сповіщення",
    userMenu: "Меню користувача",
    language: "Мова",
    switchToUk: "Перемкнути мову на українську",
    switchToEn: "Switch language to English",
  },
  en: {
    tagline: "Triathlon Intelligence",
    workspace: "Kachamba Lab",
    swimming: "Swimming",
    running: "Running",
    cycling: "Cycling",
    dryland: "Dryland",
    rehabilitation: "Rehabilitation",
    history: "History",
    assistant: "AI Assistant",
    tools: "Tools",
    athletes: "Athletes",
    search: "Search sessions and drills…",
    loading: "Loading…",
    switchAthlete: "Switch athlete",
    addAthlete: "Add athlete",
    profile: "Profile",
    settings: "Settings",
    signOut: "Sign out",
    notifications: "Notifications",
    userMenu: "User menu",
    language: "Language",
    switchToUk: "Перемкнути мову на українську",
    switchToEn: "Switch language to English",
  },
} as const;

const localeOptions: AppLocale[] = ["uk", "en"];

export function TopNav() {
  const pathname = usePathname();
  const [me, setMe] = useState<Athlete | null>(null);
  const [roster, setRoster] = useState<Athlete[]>([]);
  const [athleteOpen, setAthleteOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const defaultLocale = pathname?.startsWith("/rehabilitation") ? "uk" : "en";
  const { locale, setLocale } = useAppLocale(defaultLocale);
  const copy = shellCopy[locale];

  useEffect(() => {
    api.me().then(setMe).catch(() => undefined);
    api.listAthletes().then(setRoster).catch(() => undefined);
  }, []);

  return (
    <header className="sticky top-0 z-40 backdrop-blur-xl bg-bg/80 border-b border-white/[0.06]">
      <div className="max-w-[1600px] mx-auto px-6">
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2.5">
              <Logo size={26} />
              <div className="flex flex-col leading-none">
                <span className="text-[15px] font-bold tracking-tight">
                  SPRINT AI
                </span>
                <span className="text-[10px] text-slate-500 tracking-widest uppercase">
                  {copy.tagline}
                </span>
              </div>
            </Link>
            <div className="hidden md:flex items-center gap-1.5 text-xs">
              <span className="text-slate-400">{copy.workspace}</span>
              <ChevronRight className="w-3 h-3 text-slate-600" />
              <span className="text-slate-200 font-medium">
                {copy.athletes}
              </span>
              <ChevronRight className="w-3 h-3 text-slate-600" />
              <span className="text-slate-200 font-medium">
                {me?.name ?? "—"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button className="hidden sm:flex items-center gap-2 px-3 h-8 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] rounded-lg text-xs text-slate-400 transition-colors">
              <Search className="w-3.5 h-3.5" />
              <span>{copy.search}</span>
              <kbd className="px-1.5 py-0.5 bg-white/[0.06] rounded text-[10px] font-mono text-slate-500">
                ⌘K
              </kbd>
            </button>

            <div
              aria-label={copy.language}
              className="hidden items-center rounded-lg border border-white/[0.06] bg-white/[0.04] p-0.5 text-[10px] font-semibold sm:flex"
            >
              {localeOptions.map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-label={value === "uk" ? copy.switchToUk : copy.switchToEn}
                  aria-pressed={locale === value}
                  onClick={() => setLocale(value)}
                  className={`h-7 rounded-md px-2 transition-colors ${
                    locale === value
                      ? "bg-cyan-400 text-slate-950"
                      : "text-slate-500 hover:text-slate-200"
                  }`}
                >
                  {value === "uk" ? "UA" : "EN"}
                </button>
              ))}
            </div>

            <div className="relative">
              <button
                aria-label={copy.switchAthlete}
                onClick={() => setAthleteOpen((v) => !v)}
                className="flex items-center gap-2 px-2.5 h-8 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] rounded-lg text-xs transition-colors"
              >
                <div className="w-5 h-5 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-[10px] font-bold text-slate-900">
                  {me?.initials ?? "—"}
                </div>
                <span className="text-slate-200 font-medium hidden sm:inline">
                  {me?.name ?? copy.loading}
                </span>
                <ChevronDown className="w-3 h-3 text-slate-500" />
              </button>
              {athleteOpen ? (
                <div className="absolute right-0 top-10 w-56 bg-elevated border border-white/[0.08] rounded-xl shadow-2xl shadow-black/40 p-1 animate-fade-in">
                  <div className="px-3 py-2 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    {copy.switchAthlete}
                  </div>
                  {roster.map((a) => (
                    <button
                      key={a.id}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-slate-200 hover:bg-white/[0.05] rounded-lg transition-colors"
                    >
                      <div className="w-6 h-6 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-[10px] font-bold text-slate-900">
                        {a.initials}
                      </div>
                      <span className="flex-1 text-left">{a.name}</span>
                      {me?.id === a.id ? (
                        <Check className="w-3.5 h-3.5 text-cyan-400" />
                      ) : null}
                    </button>
                  ))}
                  <div className="border-t border-white/[0.06] mt-1 pt-1">
                    <button className="w-full flex items-center gap-2 px-3 py-2 text-sm text-slate-400 hover:bg-white/[0.05] rounded-lg transition-colors">
                      <Plus className="w-3.5 h-3.5" />
                      {copy.addAthlete}
                    </button>
                  </div>
                </div>
              ) : null}
            </div>

            <button
              aria-label={copy.notifications}
              className="w-8 h-8 flex items-center justify-center bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] rounded-lg transition-colors relative"
            >
              <Bell className="w-3.5 h-3.5 text-slate-300" />
              <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />
            </button>

            <div className="relative">
              <button
                aria-label={copy.userMenu}
                onClick={() => setUserMenuOpen((v) => !v)}
                className="w-8 h-8 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 border border-white/[0.06] flex items-center justify-center hover:border-white/[0.12] transition-colors"
              >
                <User className="w-3.5 h-3.5 text-slate-300" />
              </button>
              {userMenuOpen ? (
                <div className="absolute right-0 top-10 w-48 bg-elevated border border-white/[0.08] rounded-xl shadow-2xl shadow-black/40 p-1 animate-fade-in">
                  <div className="px-3 py-2 border-b border-white/[0.06]">
                    <p className="text-sm font-medium text-slate-100">
                      Coach Nikita
                    </p>
                    <p className="text-xs text-slate-500">
                      {me?.handle ?? ""}
                    </p>
                  </div>
                  {[
                    { icon: User, label: copy.profile },
                    { icon: Settings, label: copy.settings },
                    { icon: LogOut, label: copy.signOut },
                  ].map((item) => (
                    <button
                      key={item.label}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-slate-300 hover:bg-white/[0.05] rounded-lg transition-colors"
                    >
                      <item.icon className="w-3.5 h-3.5" />
                      {item.label}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        </div>

        <nav className="flex items-center gap-0.5 -mb-px overflow-x-auto scrollbar-none">
          {sections.map((s) => {
            const Icon = s.icon;
            const active =
              pathname === s.href || pathname?.startsWith(s.href + "/");
            return (
              <Link
                key={s.id}
                href={s.href}
                className={`relative flex items-center gap-2 px-3.5 h-10 text-xs font-medium whitespace-nowrap transition-colors duration-200 ${
                  active
                    ? "text-slate-100"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {copy[s.id]}
                {active ? (
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]" />
                ) : null}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
