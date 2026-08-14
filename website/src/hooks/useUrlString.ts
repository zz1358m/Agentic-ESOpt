import { useUrlParam } from "./useUrlParam";

const parseString = (raw: string, fallback: string) => raw || fallback;
const serializeString = (value: string) => value;

export function useUrlString(key: string, fallback: string) {
  return useUrlParam(key, fallback, parseString, serializeString);
}
