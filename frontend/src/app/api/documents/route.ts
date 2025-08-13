import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

async function checkAuth(request: NextRequest) {
  const authToken = request.cookies.get("auth_token");
  const userEmail = request.cookies.get("user_email");
  
  if (!authToken || !userEmail) {
    return null;
  }
  
  return {
    email: userEmail.value,
    username: userEmail.value.split('@')[0],
    is_admin: request.cookies.get("is_admin")?.value === "true"
  };
}

export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const user = await checkAuth(request);
    if (!user) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    const response = await fetch(`${BACKEND_URL}/api/documents`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer authenticated",
        "X-User-Email": user.email,
        "X-User-Admin": user.is_admin.toString(),
        ...(request.headers.get("cookie") && {
          Cookie: request.headers.get("cookie")!,
        }),
      },
    });

    const data = await response.json();
    
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to fetch documents" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const user = await checkAuth(request);
    if (!user || !user.is_admin) {
      return NextResponse.json(
        { error: user ? "Admin access required" : "Authentication required" },
        { status: user ? 403 : 401 }
      );
    }

    const formData = await request.formData();
    
    const response = await fetch(`${BACKEND_URL}/api/documents/upload`, {
      method: "POST",
      body: formData,
      headers: {
        "Authorization": "Bearer authenticated",
        "X-User-Email": user.email,
        "X-User-Admin": user.is_admin.toString(),
        ...(request.headers.get("cookie") && {
          Cookie: request.headers.get("cookie")!,
        }),
      },
    });

    const data = await response.json();
    
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to upload document" },
      { status: 500 }
    );
  }
}
