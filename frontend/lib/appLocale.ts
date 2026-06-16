"use client";

import { useCallback, useEffect, useState } from "react";

import {
  REHAB_LOCALE_STORAGE_KEY,
  type RehabLocale,
} from "@/lib/rehabCopy";

export type AppLocale = RehabLocale;

export const APP_LOCALE_STORAGE_KEY = REHAB_LOCALE_STORAGE_KEY;
export const APP_LOCALE_CHANGE_EVENT = "rehab-locale-change";

export function isAppLocale(value: unknown): value is AppLocale {
  return value === "uk" || value === "en";
}

export function useAppLocale(defaultLocale: AppLocale = "en") {
  const [locale, setLocaleState] = useState<AppLocale>(defaultLocale);

  useEffect(() => {
    const stored = window.localStorage.getItem(APP_LOCALE_STORAGE_KEY);
    setLocaleState(isAppLocale(stored) ? stored : defaultLocale);
  }, [defaultLocale]);

  useEffect(() => {
    document.documentElement.lang = locale === "uk" ? "uk" : "en";
  }, [locale]);

  useEffect(() => {
    const onLocaleChange = (event: Event) => {
      const value = (event as CustomEvent<AppLocale>).detail;
      if (isAppLocale(value)) setLocaleState(value);
    };
    window.addEventListener(APP_LOCALE_CHANGE_EVENT, onLocaleChange);
    return () =>
      window.removeEventListener(APP_LOCALE_CHANGE_EVENT, onLocaleChange);
  }, []);

  const setLocale = useCallback((value: AppLocale) => {
    window.localStorage.setItem(APP_LOCALE_STORAGE_KEY, value);
    setLocaleState(value);
    window.dispatchEvent(
      new CustomEvent<AppLocale>(APP_LOCALE_CHANGE_EVENT, { detail: value }),
    );
  }, []);

  return { locale, setLocale };
}
