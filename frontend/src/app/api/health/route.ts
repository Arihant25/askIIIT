import { NextResponse } from "next/server";
import ApiService from "@/lib/api";

export async function GET() {
  try {
    // Check backend health by calling the backend API
    await ApiService.getHealth();
    return NextResponse.json({ status: "healthy" }, { status: 200 });
  } catch (error) {
    console.error("Health check error:", error);
    return NextResponse.json(
      { status: "unhealthy", error: "Backend is not available" },
      { status: 503 }
    );
  }
}
