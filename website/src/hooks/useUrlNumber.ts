import { useUrlParam } from "./useUrlParam";

const parseNumber = (raw: string, fallback: number) => Number.isNaN(Number(raw)) ? fallback : Number(raw);
const serializeNumber = (value: number) => String(value);

export function useUrlNumber(key: string, fallback: number) {
  return useUrlParam(key, fallback, parseNumber, serializeNumber);
}
