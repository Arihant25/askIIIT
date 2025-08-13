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

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    // Check authentication
    const user = await checkAuth(request);
    if (!user) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    const response = await fetch(`${BACKEND_URL}/api/documents/${params.id}/download`, {
      method: "GET",
      headers: {
        "Authorization": "Bearer authenticated",
        "X-User-Email": user.email,
        "X-User-Admin": user.is_admin.toString(),
        ...(request.headers.get("cookie") && {
          Cookie: request.headers.get("cookie")!,
        }),
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to download document" },
        { status: response.status }
      );
    }

    const blob = await response.blob();
    const filename = response.headers.get('Content-Disposition')?.split('filename=')?.[1]?.replace(/"/g, '') || 'document.pdf';
    
    return new NextResponse(blob, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${filename}"`,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to download document" },
      { status: 500 }
    );
  }
}