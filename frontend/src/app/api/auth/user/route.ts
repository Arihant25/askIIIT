import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    // Get user data from cookies that were set during authentication
    const authToken = request.cookies.get("auth_token");
    const userEmail = request.cookies.get("user_email");
    const userName = request.cookies.get("user_name");
    const isAdmin = request.cookies.get("is_admin");

    console.log("Auth check - cookies found:", {
      authToken: !!authToken,
      userEmail: userEmail?.value,
      userName: userName?.value,
      isAdmin: isAdmin?.value
    });

    if (!authToken || !userEmail) {
      console.log("No valid authentication found - missing token or email");
      return NextResponse.json(
        { error: "Not authenticated" },
        { status: 401 }
      );
    }

    // Return user info from cookies
    const userInfo = {
      email: userEmail.value,
      name: userName?.value || userEmail.value.split('@')[0],
      full_name: userName?.value || userEmail.value.split('@')[0],
      username: userEmail.value.split('@')[0],
      is_admin: isAdmin?.value === "true"
    };
    
    console.log("Returning user info from cookies:", userInfo);
    return NextResponse.json(userInfo);
  } catch (error) {
    console.error("Auth user check error:", error);
    return NextResponse.json(
      { error: "Failed to get user info" },
      { status: 500 }
    );
  }
}
