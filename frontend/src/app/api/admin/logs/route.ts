import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

async function checkAdminAuth(request: NextRequest) {
  const authToken = request.cookies.get("auth_token");
  const userEmail = request.cookies.get("user_email");
  const isAdmin = request.cookies.get("is_admin")?.value === "true";
  
  if (!authToken || !userEmail || !isAdmin) {
    return false;
  }
  
  return true;
}

export async function GET(request: NextRequest) {
  try {
    // Check admin authentication
    const isAuthorized = await checkAdminAuth(request);
    if (!isAuthorized) {
      return NextResponse.json(
        { error: "Admin access required" },
        { status: 403 }
      );
    }

    const response = await fetch(`${BACKEND_URL}/api/admin/logs`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer authenticated",
        ...(request.headers.get("cookie") && {
          Cookie: request.headers.get("cookie")!,
        }),
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { logs: [] },
        { status: 200 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { logs: [] },
      { status: 200 }
    );
  }
}